
# -- python --
import torch as th
import numpy as np
from einops import rearrange

# -- cpp cuda kernel --
import stnls_cuda

# -- package --
import stnls

# -- local --
from stnls.search.utils import allocate_pair,dist_type_select,allocate_vid
from stnls.search.utils import get_inds,allocate_grad_flows
from stnls.search.shared import normz_bwd

def forward(vid0, vid1, flows,
            ws, wr, k, kr, ps,
            stride0, stride1, strideQ, dilation, pt,
            dist_type, restricted_radius,
            reflect_bounds, full_ws,
            topk_mode, self_action, patch_offset,
            off_Hq, off_Wq, itype_fwd):

    # -- fix negative Q --
    # if Q > 0:
    #     flows = flows[:,:,qshift:qshift+Q].contiguous()
    B,HD,T,C,qH,qW = vid0.shape
    B,HD,T,C,kH,kW = vid1.shape
    B,HD,T,nH,nW,Ks,_ = flows.shape
    Q = T*nH*nW

    # -- settings from distance type --
    dist_type_i,descending,idist_val = dist_type_select(dist_type)

    # -- allocate results --
    device = flows.device
    B,HD,T,nH,nW,Ks = flows.shape[:-1]
    base_shape = (B,HD,T,nH,nW,Ks,wr,wr)
    # print(base_shape,flows.shape)
    dists,inds = allocate_pair(base_shape,device,vid0.dtype,idist_val,itype_fwd)

    # -- allow for int fwd when actually float --
    if itype_fwd == "int":
        inds = inds.int()
        if flows.dtype == th.float:
            flows = flows.round().int()
        kselect = th.zeros(0,device=flows.device)
        reflect = th.zeros(0,device=flows.device)
    else:
        kselect = th.ones_like(dists).int()
        reflect = th.zeros_like(flows[...,:2]).bool()

    # -- run --
    if itype_fwd == "int":
        stride1 = int(max(1,int(stride1)))
        fwd_fxn = stnls_cuda.refinement_int_forward
        strideQ = stride0 if strideQ is None else strideQ
        fwd_fxn(vid0, vid1, flows, dists, inds,
                ws, ps, stride0, stride1, strideQ, dilation, pt,
                restricted_radius, reflect_bounds, full_ws,
                patch_offset, off_Hq, off_Wq, dist_type_i)
    else:
        stride1 = float(stride1)
        fwd_fxn = stnls_cuda.refinement_bilin2d_forward
        fwd_fxn(vid0, vid1, flows, dists, inds,
                kselect, reflect,
                ws, ps, stride0, stride1, dilation, pt,
                restricted_radius, reflect_bounds, full_ws,
                patch_offset, off_Hq, off_Wq, dist_type_i)

    # -- manage self dists --
    if not(self_action is None) and "anchor" in self_action:
        H,W = vid0.shape[-2:]
        stnls.nn.anchor_self_refine(dists,inds,flows,stride0,qH,qW,kH,kW)
    else:
        assert self_action == None

    # -- topk --
    assert self_action in [None,"anchor","anchor_self","anchor_each"]
    anchor_self = False if self_action is None else "anchor" in self_action
    if topk_mode == "all":
        dim = 3
        dists=dists.view(B,HD,Q,Ks*wr*wr)
        inds=inds.view(B,HD,Q,Ks*wr*wr,3)
        dists,inds,order = stnls.nn.topk(dists,inds,k,dim=dim,anchor=anchor_self,
                                         descending=descending,unique=False,
                                         return_order=True)
        # print(order)
        if kselect.ndim > 1:
            # print("kselect.shape: ",kselect.shape,order.shape)
            kselect = kselect.view(B,HD,Q,Ks*wr*wr) if not(kselect is None) else kselect
            # print("kselect.shape: ",kselect.shape,order.shape)
            kselect = stnls.nn.topk_f.apply_topk(kselect,order,dim)
    elif topk_mode == "each":
        # print(dists.shape,kselect.shape)
        dists = rearrange(dists,'... wh ww -> ... (wh ww)')
        inds = rearrange(inds,'... wh ww d2or3 -> ... (wh ww) d2or3')
        dists,inds = stnls.nn.topk_each(dists,inds,k,descending,anchor_self=anchor_self)
        if kselect.ndim > 1:
            kselect = rearrange(kselect,'... wh ww -> ... (wh ww)')
            kselect = kselect[...,:k] # all same across dim
    else:
        raise ValueError(f"Unknown topk_mode [{topk_mode}]")


    # -- reshape for output --
    dists=dists.view(B,HD,T,nH,nW,-1)
    inds=inds.view(B,HD,T,nH,nW,-1,3)
    kselect = kselect.view(B,HD,T,nH,nW,-1) if not(kselect is None) else kselect
    # print("kselect.shape,reflect.shape: ",kselect.shape,reflect.shape)
    # print(flows.shape,inds.shape,kselect.shape)
    # print(th.cat([flows[0,0,...,[0]],inds[0,0,...,[0]],kselect[0,0,...,None]],-1))

    return dists,inds,kselect,reflect

def backward(ctx, grad_dists, grad_inds):

    # -- populate names --
    inds,vid0,vid1,kselect,reflect = ctx.saved_tensors
    itype_bwd = ctx.itype_bwd
    device = grad_dists.device

    # -- allocate grads --
    grad_vid0 = allocate_vid(ctx.vid_shape,device)
    grad_vid1 = allocate_vid(ctx.vid_shape,device)
    grad_flows = allocate_grad_flows(itype_bwd,ctx.flows_shape,device)

    # -- restrict to k_agg --
    if ctx.k_agg > 0:
        grad_dists = grad_dists[...,:ctx.k_agg]
        inds = inds[...,:ctx.k_agg,:]

    # -- ensure contiguous --
    grad_dists = grad_dists.contiguous()
    inds = get_inds(inds,itype_bwd)
    patch_offset = 0 if ctx.use_adj else -(ctx.ps//2)

    # -- backward pass with increasing complexity --
    # print(inds[...,1:].min().item(),inds[...,1:].max().item())
    # print("gvid0: ",grad_vid0.shape)
    # print(kselect.min().item(),kselect.max().item())
    if itype_bwd == "int":
        strideQ = ctx.stride0 if ctx.strideQ is None else ctx.strideQ
        bwd_fxn = stnls_cuda.non_local_search_int_vid_backward
        bwd_fxn(grad_vid0,grad_vid1,vid0,vid1,grad_dists,inds,
                ctx.ps,ctx.pt,ctx.stride0,strideQ,ctx.dil,
                ctx.reflect_bounds,patch_offset,
                ctx.off_Hq, ctx.off_Wq, ctx.dist_type_i)
    elif not(ctx.flows_requires_grad):
        bwd_fxn = stnls_cuda.non_local_search_bilin2d_vid_backward
        bwd_fxn(grad_vid0,grad_vid1,vid0,vid1,grad_dists,inds,
                ctx.wt,ctx.ps,ctx.pt,ctx.stride0,ctx.dil,
                ctx.reflect_bounds,patch_offset,
                ctx.off_Hq, ctx.off_Wq, ctx.dist_type_i)
    else:
        bwd_fxn = stnls_cuda.refinement_bilin2d_vidflows_backward
        bwd_fxn(grad_vid0,grad_vid1,grad_flows,
                vid0,vid1,grad_dists,grad_inds,inds,
                kselect,reflect,
                ctx.ws,ctx.ps,ctx.pt,ctx.stride0,ctx.dil,
                ctx.reflect_bounds,patch_offset,
                ctx.off_Hq, ctx.off_Wq, ctx.dist_type_i)
    th.cuda.synchronize()

    # -- finalize shape --
    grad_vid0 = rearrange(grad_vid0,'B H t c h w -> B t (H c) h w')
    grad_vid1 = rearrange(grad_vid1,'B H t c h w -> B t (H c) h w')

    # -- normz --
    if ctx.normalize_bwd:
        normz_bwd(ctx,grad_vid0,grad_vid1)

    # -- no grad if ints --
    if itype_bwd == "int": grad_flows = None

    return grad_vid0,grad_vid1,grad_flows

