# -- python --
import torch as th
import numpy as np
from einops import rearrange

# -- cpp cuda kernel --
import stnls_cuda

# -- package --
import stnls

# -- api --
from stnls.utils import extract_pairs

# -- forward utils --
from stnls.search.utils import allocate_pair,dist_type_select,allocate_vid

# -- backward utils --
from stnls.search.utils import get_inds,allocate_grad_flows
from stnls.search.shared import normz_bwd


def forward(vid0, vid1, flows,
            ws, wt, ps, k, stride0, stride1,
            dist_type, dilation, pt,
            topk_mode, self_action,
            reflect_bounds, full_ws, use_adj,
            off_Hq, off_Wq, itype):

    # -- unpack --
    # itype = "int"
    device = vid0.device
    B,HD,T,C,qH,qW = vid0.shape
    B,HD,T,C,kH,kW = vid1.shape
    patch_offset = 0 if use_adj else -(ps//2)
    # print(ps,k,dist_type,topk_mode,self_action,patch_offset)

    # -- derived shapes --
    nH0 = (kH-1)//stride0+1
    nW0 = (kW-1)//stride0+1
    Q = T*nH0*nW0
    # print(vid0.shape,nH0,nW0,Q)

    # -- settings from distance type --
    dist_type_i,descending,idist_val = dist_type_select(dist_type)

    # -- allocate results --
    W_t = min(2*wt+1,T)
    base_shape = (B,HD,Q,W_t,ws,ws)
    dists,inds = allocate_pair(base_shape,device,vid0.dtype,idist_val,itype)

    # -- check flows --
    assert flows.shape[3] in [W_t-1,W_t]

    # -- forward --
    if itype == "int":
        if flows.dtype != th.int:
            flows = flows.round().int()
        else:
            flows = flows.int()
        inds = inds.int()
        stride1 = max(1,int(stride1))
        fwd_fxn = stnls_cuda.non_local_search_int_forward
    else:
        fwd_fxn = stnls_cuda.non_local_search_bilin2d_forward
        stride1 = float(stride1)
    fwd_fxn(vid0, vid1, flows, dists, inds,
            ps, k, stride0, stride1, dilation, pt,
            reflect_bounds, full_ws, patch_offset,
            off_Hq, off_Wq, dist_type_i)

    # -- anchor --
    menu = [None,"anchor","anchor_self","anchor_each","remove",]
    menu += ["remove_ref_frame","anchor_and_remove_ref_frame"]
    assert self_action in menu
    anchor_self = False if self_action is None else "anchor" in self_action
    if self_action in ["anchor","anchor_self"]:
        stnls.nn.anchor_self(dists,inds,stride0,nH0,nW0)
    elif self_action == "anchor_each":
        stnls.nn.anchor_self_time(dists,inds,flows,wt,stride0,qH,qW,kH,kW)
    elif self_action == "remove":
        stnls.nn.anchor_self(dists,inds,stride0,nH0,nW0)
        dists=dists.view(B,HD,Q,-1)
        inds=inds.view(B,HD,Q,-1,3)
        dists = dists[:,:,:,1:].contiguous()
        inds = inds[:,:,:,1:,:].contiguous()
    elif self_action == "remove_ref_frame":
        assert wt > 0,"Cannot remove ref frame if not searching across time."
        dists = dists[...,1:,:,:].contiguous()
        inds = inds[...,1:,:,:,:].contiguous()
    elif self_action == "anchor_and_remove_ref_frame":
        assert wt > 0,"Cannot remove ref frame if not searching across time."
        dists = dists[...,1:,:,:].contiguous()
        inds = inds[...,1:,:,:,:].contiguous()
        stnls.nn.anchor_self_time(dists,inds,flows,wt,stride0,qH,qW,kH,kW)
    elif self_action is None:
        pass
    else:
        raise ValueError(f"Uknown option for self_action [{self_action}]")

    # -- topk --
    if topk_mode == "all":
        dim = 3
        dists=dists.view(B,HD,Q,-1)
        inds=inds.view(B,HD,Q,-1,3)
        # dists=dists.view(B,HD,Q,W_t*ws*ws)
        # inds=inds.view(B,HD,Q,W_t*ws*ws,3)
        dists,inds = stnls.nn.topk(dists,inds,k,dim=dim,anchor=anchor_self,
                                   descending=descending)
    elif topk_mode == "each":
        dists = rearrange(dists,'... wh ww -> ... (wh ww)')
        inds = rearrange(inds,'... wh ww d2or3 -> ... (wh ww) d2or3')
        dists,inds = stnls.nn.topk_each(dists,inds,k,descending,anchor_self=anchor_self)
    else:
        raise ValueError(f"Unknown topk_mode [{topk_mode}]")

    # -- reshape --
    dists=dists.view(B,HD,T,nH0,nW0,-1)
    inds=inds.view(B,HD,T,nH0,nW0,-1,3)

    return dists,inds

def backward(ctx, grad_dists, grad_inds):

    # -- populate names --
    dists,inds,vid0,vid1,flows = ctx.saved_tensors
    itype_bwd = ctx.itype

    # -- allocate grads --
    grad_vid0 = allocate_vid(ctx.vid_shape,grad_dists.device)
    grad_vid1 = allocate_vid(ctx.vid_shape,grad_dists.device)
    grad_flows = allocate_grad_flows(itype_bwd,flows.shape,flows.device)

    # -- restrict to k_agg; the number of neighbors used which will prop gradient --
    if ctx.k_agg > 0:
        grad_dists = grad_dists[...,:ctx.k_agg].contiguous()
        grad_inds = grad_inds[...,:ctx.k_agg].contiguous()
        dists = dists[...,:ctx.k_agg].contiguous()
        inds = inds[...,:ctx.k_agg,:]
    dists = dists.contiguous()
    inds = inds.contiguous()

    # -- ensure contiguous & type --
    inds = get_inds(inds,ctx.itype)
    patch_offset = 0 if ctx.use_adj else -(ctx.ps//2)
    reflect_bounds = ctx.reflect_bounds

    # -- backward pass with increasing complexity --
    if ctx.itype == "int":
        bwd_fxn = stnls_cuda.non_local_search_int_vid_backward
        bwd_fxn(grad_vid0,grad_vid1,
                vid0,vid1,grad_dists,inds,
                ctx.ps,ctx.pt,ctx.stride0,ctx.dil,
                reflect_bounds,patch_offset,
                ctx.off_Hq,ctx.off_Wq,ctx.dist_type_i)
    elif not(flows.requires_grad):
        bwd_fxn = stnls_cuda.non_local_search_bilin2d_vid_backward
        bwd_fxn(grad_vid0,grad_vid1,vid0,vid1,
                grad_dists,inds,
                ctx.wt,ctx.ps,ctx.pt,ctx.stride0,ctx.dil,
                reflect_bounds,patch_offset,
                ctx.off_Hq, ctx.off_Wq, ctx.dist_type_i)
    else:
        bwd_fxn = stnls_cuda.non_local_search_bilin2d_vidflows_backward
        bwd_fxn(grad_vid0,grad_vid1,grad_flows,
                vid0,vid1,flows,
                grad_dists,grad_inds,dists,inds,
                ctx.wt,ctx.ps,ctx.pt,ctx.stride0,ctx.dil,
                reflect_bounds,patch_offset,
                ctx.off_Hq, ctx.off_Wq, ctx.dist_type_i)

    # -- finalize shape --
    grad_vid0 = rearrange(grad_vid0,'B H t c h w -> B t (H c) h w')
    grad_vid1 = rearrange(grad_vid1,'B H t c h w -> B t (H c) h w')

    # -- normz --
    if ctx.normalize_bwd:
        normz_bwd(ctx,grad_vid0,grad_vid1)

    # -- no grad if ints --
    if itype_bwd == "int" or not(flows.requires_grad):
        grad_flows = None
    if ctx.flow_ndim == 6 and flows.requires_grad:
        grad_flows = grad_flows[:,0].contiguous()
    # print(grad_flows.shape)
    # print(th.where(flows[0,0]!=0))
    # print(th.where(grad_flows[0,0]!=0))
    # print("-"*20)
    # print(th.all(grad_flows==0).item())

    return grad_vid0,grad_vid1,grad_flows

