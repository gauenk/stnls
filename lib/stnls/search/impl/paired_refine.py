
# -- python --
import torch as th
import numpy as np
from einops import rearrange

# -- cpp cuda kernel --
import stnls_cuda

# -- package --
import stnls

# -- local --
from stnls.search.utils import shape_frames,allocate_pair_2d,dist_type_select,allocate_vid
from stnls.search.utils import get_ctx_shell,ensure_flow_shape
from stnls.search.utils import ensure_paired_flow_dim as ensure_flow_dim
from stnls.search.shared import manage_self,reflect_bounds_warning
from stnls.search.utils import paired_vids as _paired_vids
from stnls.search.utils import shape_vids,allocate_pair,dist_type_select,allocate_vid
from stnls.search.utils import get_inds,allocate_grad_flows
from stnls.search.shared import manage_self

def forward(frame0, frame1, flow,
            ws, wr, k, ps, nheads, dist_type,
            stride0, stride1, dilation, self_action,
            restricted_radius, reflect_bounds, full_ws,
            use_adj, topk_mode, itype):

    # -- unpack --
    device = frame0.device
    B,HD_fr,C,qH,qW = frame0.shape
    kH,kW = frame0.shape[-2:]
    HD_flow = flow.shape[1]
    # print("flow.shape: ",flow.shape)
    # print(frame0.shape,flow.shape)
    assert flow.ndim == 6
    HD = max(HD_flow,HD_fr)
    patch_offset = 0 if use_adj else -(ps//2)
    B,HD,nH_fl,nW_fl,Ks,_ = flow.shape

    # -- derived shapes --
    nH0 = (kH-1)//stride0+1
    nW0 = (kW-1)//stride0+1
    Q = nH0*nW0

    # -- settings from distance type --
    dist_type_i,descending,idist_val = dist_type_select(dist_type)

    # -- allocate results --
    base_shape = (B,HD,Q,Ks,wr,wr)
    dists,inds = allocate_pair_2d(base_shape,device,frame0.dtype,idist_val,itype)
    # print("inds.shape: ",inds.shape)

    # -- forward --
    if itype == "int":
        flow = flow.round()
        inds = inds.int()
        stride1 = max(1,int(stride1))
        kselect = th.zeros(0,device=flow.device)
        fwd_fxn = stnls_cuda.paired_refine_int_forward
    else:
        kselect = th.ones_like(dists).int()
        fwd_fxn = stnls_cuda.paired_refine_bilin2d_forward
        stride1 = float(stride1)

    # -- run --
    if itype == "int":
        fwd_fxn(frame0, frame1, flow, dists, inds,
                ws, ps, stride0, stride1, dilation,
                restricted_radius, reflect_bounds, full_ws,
                patch_offset, dist_type_i)
        flow = flow.int()
    else:
        fwd_fxn(frame0, frame1, flow, dists, inds,
                kselect, ws, ps, stride0, stride1, dilation,
                restricted_radius, reflect_bounds, full_ws,
                patch_offset, dist_type_i)

    # # print(frame0.shape,flow.shape,dists.shape,inds.shape)
    # fwd_fxn(frame0, frame1, flow, dists, inds,
    #         ws, ps, k, stride0, stride1, dilation,
    #         reflect_bounds, full_ws, patch_offset, dist_type_i)


    # -- anchor --
    assert self_action in [None,"anchor","anchor_each"]
    anchor_self = False if self_action is None else "anchor" in self_action
    if self_action is None: pass
    elif "anchor" in self_action:
        # print(dists.shape,inds.shape,flow.shape)
        stnls.nn.anchor_self_paired(dists,inds,flow,stride0,qH,qW,kH,kW)
    else:
        raise ValueError(f"Uknown option for self_action [{self_action}]")

    # print(frame0.shape,frame1.shape,flow.shape,inds.shape)
    # -- compress search region --
    # dists=dists.view(B,HD,Q,-1)
    # inds=inds.view(B,HD,Q,-1,2)
    # # th.cuda.synchronize()
    # print("[pre]: ",dists.shape,inds.shape,topk_mode,anchor_self)

    # -- topk --
    assert self_action in [None,"anchor","anchor_each"]
    anchor_self = False if self_action is None else "anchor" in self_action
    if topk_mode == "all" and (k > 0):
        dim = 3
        dists=dists.view(B,HD,Q,-1)
        inds=inds.view(B,HD,Q,-1,2)
        dists,inds,order = stnls.nn.topk(dists,inds,k,dim=dim,anchor=anchor_self,
                                         descending=descending,unique=False,
                                         return_order=True)
        if kselect.ndim > 1:
            # print("kselect.shape: ",kselect.shape,order.shape)
            kselect = kselect.view(B,HD,Q,Ks*wr*wr) if not(kselect is None) else kselect
            # print("kselect.shape: ",kselect.shape,order.shape)
            kselect = stnls.nn.topk_f.apply_topk(kselect,order,dim)
    elif topk_mode == "each" and (k > 0):
        dists = rearrange(dists,'... wh ww -> ... (wh ww)')
        inds = rearrange(inds,'... wh ww d2or3 -> ... (wh ww) d2or3')
        dists,inds = stnls.nn.topk_each(dists,inds,k,descending,anchor_self=anchor_self)
        if kselect.ndim > 1 and k > 0:
            kselect = rearrange(kselect,'... wh ww -> ... (wh ww)')
            kselect = kselect[...,:k] # all same across dim
    elif (k > 0):
        raise ValueError(f"Unknown topk_mode [{topk_mode}]")
    # print("[post]: ",dists.shape,inds.shape)

    # # -- topk --
    # if k > 0:
    #     dim = 3
    #     dists=dists.view(B,HD,Q,Ks*wr*wr)
    #     inds=inds.view(B,HD,Q,Ks*wr*wr,2)
    #     dists,inds = stnls.nn.topk(dists,inds,k,dim=dim,anchor=anchor_self,
    #                                descending=descending)

    # -- reshape --
    dists=dists.reshape(B,HD,nH0,nW0,-1)
    inds=inds.reshape(B,HD,nH0,nW0,-1,2)
    if kselect.ndim > 1:
        kselect = kselect.reshape(B,HD,nH0*nW0,-1)
        assert kselect.numel() == dists.numel()

    return dists,inds,kselect


def backward(ctx, grad_dists, grad_inds):

    # -- populate names --
    inds,frame0,frame1,flow,kselect = ctx.saved_tensors
    itype_bwd = ctx.itype
    inds = get_inds(inds,itype_bwd)
    grad_flow = allocate_grad_flows(itype_bwd,flow.shape,flow.device)

    # -- allocate grads --
    grad_frame0 = allocate_vid(frame0.shape,grad_dists.device)
    grad_frame1 = allocate_vid(frame1.shape,grad_dists.device)

    # -- restrict to k_agg --
    if ctx.k_agg > 0:
        grad_dists = grad_dists[...,:ctx.k_agg]
        inds = inds[...,:ctx.k_agg,:]

    # -- ensure contiguous --
    grad_dists = grad_dists.contiguous()
    inds = inds.contiguous()

    # -- view --
    B,HD,nH,nW,K,_ = inds.shape
    inds = inds.view(B,HD,nH*nW,K,2)
    grad_inds = grad_inds.view(B,HD,nH*nW,K,2)
    grad_dists = grad_dists.view(B,HD,nH*nW,K)
    patch_offset = 0 if ctx.use_adj else -(ctx.ps//2)

    # -- allow for repeated exec --
    grad_inds = grad_inds.contiguous()
    if itype_bwd == "int":
        bwd_fxn = stnls_cuda.paired_search_int_backward
        inds = inds.view(B,HD,nH*nW,K,2)
        bwd_fxn(grad_frame0,grad_frame1,
                frame0,frame1,grad_dists,inds,
                ctx.stride0,ctx.ps,ctx.dil,ctx.reflect_bounds,
                patch_offset,ctx.dist_type_i)
    else:
        # print("grad_flow.shape,flow.shape: ",grad_flow.shape,flow.shape)
        bwd_fxn = stnls_cuda.paired_refine_vidflows_backward
        bwd_fxn(grad_frame0,grad_frame1,grad_flow,
                frame0,frame1,flow,grad_dists,grad_inds,inds,
                kselect,ctx.stride0,ctx.ps,ctx.dil,ctx.reflect_bounds,
                patch_offset,ctx.dist_type_i)

    # -- finalize shape --
    if ctx.in_ndim == 4:
        grad_frame0 = rearrange(grad_frame0,'B H c h w -> B (H c) h w')
        grad_frame1 = rearrange(grad_frame1,'B H c h w -> B (H c) h w')

    # -- normz --
    if ctx.normalize_bwd:
        normz_bwd(ctx,grad_frame0,grad_frame1)

    # -- no grad if ints --
    if itype_bwd == "int":
        grad_flow = None

    return grad_frame0,grad_frame1,grad_flow

