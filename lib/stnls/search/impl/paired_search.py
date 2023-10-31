
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
            ws, ps, k, dist_type,
            stride0, stride1, dilation, pt,
            self_action, reflect_bounds,
            full_ws, use_adj, itype):

    # -- unpack --
    device = frame0.device
    B,HD_fr,C,H,W = frame0.shape
    HD_flow = flow.shape[1]
    # print(frame0.shape,flow.shape)
    assert flow.ndim == 5
    HD = max(HD_flow,HD_fr)
    patch_offset = 0 if use_adj else -(ps//2)

    # -- derived shapes --
    nH0 = (H-1)//stride0+1
    nW0 = (W-1)//stride0+1
    Q = nH0*nW0

    # -- settings from distance type --
    dist_type_i,descending,idist_val = dist_type_select(dist_type)

    # -- allocate results --
    base_shape = (B,HD,Q,ws,ws)
    dists,inds = allocate_pair_2d(base_shape,device,frame0.dtype,idist_val,itype)
    # print("inds.shape: ",inds.shape)

    # -- forward --
    if itype == "int":
        flow = flow.round().int()
        inds = inds.int()
        stride1 = max(1,int(stride1))
        fwd_fxn = stnls_cuda.paired_search_int_forward
    else:
        fwd_fxn = stnls_cuda.paired_search_bilin2d_forward
        stride1 = float(stride1)
    fwd_fxn(frame0, frame1, flow, dists, inds,
            ps, k, stride0, stride1, dilation,
            reflect_bounds, full_ws, patch_offset, dist_type_i)

    # -- anchor --
    assert self_action in [None,"anchor","anchor_each"]
    anchor_self = False if self_action is None else "anchor" in self_action
    if self_action is None: pass
    elif "anchor" in self_action:
        dists,inds = dists[...,None,:,:],inds[...,None,:,:,:]
        flow = rearrange(flow,'b hd two nh nw -> b hd nh nw 1 two').flip(-1)
        flow = flow.contiguous()
        # print("dists.shape,inds.shape,flow.shape: ",dists.shape,inds.shape,flow.shape)
        stnls.nn.anchor_self_paired(dists,inds,flow,stride0,H,W)
    else:
        raise ValueError(f"Uknown option for self_action [{self_action}]")

    # -- topk --
    if k > 0:
        dim = 3
        dists=dists.view(B,HD,Q,-1)
        inds=inds.view(B,HD,Q,-1,2)
        dists,inds = stnls.nn.topk(dists,inds,k,dim=dim,anchor=anchor_self,
                                   descending=descending)

    # -- reshape --
    dists=dists.reshape(B,HD,nH0,nW0,-1)
    inds=inds.reshape(B,HD,nH0,nW0,-1,2)

    return dists,inds



def backward(ctx, grad_dists, grad_inds):

    # -- populate names --
    inds,frame0,frame1,flow = ctx.saved_tensors
    itype_bwd = ctx.itype
    inds = get_inds(inds,itype_bwd)
    grad_flow = allocate_grad_flows(itype_bwd,flow.shape,flow.device)

    # -- allocate grads --
    grad_frame0 = allocate_vid(ctx.vid_shape,grad_dists.device)
    grad_frame1 = allocate_vid(ctx.vid_shape,grad_dists.device)
    # return grad_vid0,grad_vid1,grad_flow

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
        bwd_fxn = stnls_cuda.paired_search_bilin2d_backward
        bwd_fxn(grad_frame0,grad_frame1,grad_flow,
                frame0,frame1,flow,grad_dists,grad_inds,inds,
                ctx.stride0,ctx.ps,ctx.dil,ctx.reflect_bounds,
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

