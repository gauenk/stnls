
# -- python --
import torch as th
import numpy as np
from einops import rearrange

# -- cpp cuda kernel --
import stnls_cuda

# -- package --
import stnls

# -- local --
from .utils import shape_vids,allocate_pair,dist_type_select,allocate_vid
from .utils import get_inds,allocate_grad_flows
from .shared import manage_self

def paired_backward(ctx, grad_dists, grad_inds):

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
    B,HD,T,nH,nW,K,_ = inds.shape
    inds = inds.view(B,HD,T*nH*nW,K,2)
    grad_inds = grad_inds.view(B,HD,T*nH*nW,K,2)
    grad_dists = grad_dists.view(B,HD,T*nH*nW,K)
    patch_offset = 0 if ctx.use_adj else -(ctx.ps//2)

    # -- allow for repeated exec --
    grad_inds = grad_inds.contiguous()
    if itype_bwd == "int":
        bwd_fxn = stnls_cuda.paired_search_int_backward
        inds = inds.view(B,HD,T*nH*nW,K,2)
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


def paired_refine_backward(ctx, grad_dists, grad_inds):

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
    B,HD,T,nH,nW,K,_ = inds.shape
    inds = inds.view(B,HD,T*nH*nW,K,2)
    grad_inds = grad_inds.view(B,HD,T*nH*nW,K,2)
    grad_dists = grad_dists.view(B,HD,T*nH*nW,K)
    patch_offset = 0 if ctx.use_adj else -(ctx.ps//2)

    # -- allow for repeated exec --
    grad_inds = grad_inds.contiguous()
    if itype_bwd == "int":
        bwd_fxn = stnls_cuda.paired_refine_int_backward
        inds = inds.view(B,HD,T*nH*nW,K,2)
        bwd_fxn(grad_frame0,grad_frame1,
                frame0,frame1,grad_dists,inds,
                ctx.stride0,ctx.ps,ctx.dil,ctx.reflect_bounds,
                patch_offset,ctx.dist_type_i)
    else:
        bwd_fxn = stnls_cuda.paired_refine_bilin2d_backward
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


