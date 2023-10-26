
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
from .shared import manage_self,normz_bwd

def ref_backward(ctx, grad_dists, grad_inds):

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
        bwd_fxn = stnls_cuda.non_local_search_int_vid_backward
        bwd_fxn(grad_vid0,grad_vid1,vid0,vid1,grad_dists,inds,
                ctx.ps,ctx.pt,ctx.stride0,ctx.dil,
                ctx.reflect_bounds,patch_offset,ctx.dist_type_i)
    elif not(ctx.flows_requires_grad):
        bwd_fxn = stnls_cuda.non_local_search_bilin2d_vid_backward
        bwd_fxn(grad_vid0,grad_vid1,vid0,vid1,grad_dists,inds,
                ctx.wt,ctx.ps,ctx.pt,ctx.stride0,ctx.dil,
                ctx.reflect_bounds,patch_offset,ctx.dist_type_i)
    else:
        bwd_fxn = stnls_cuda.refinement_bilin2d_vidflows_backward
        bwd_fxn(grad_vid0,grad_vid1,grad_flows,
                vid0,vid1,grad_dists,grad_inds,inds,
                kselect,reflect,
                ctx.ws,ctx.ps,ctx.pt,ctx.stride0,ctx.dil,
                ctx.reflect_bounds,patch_offset,ctx.dist_type_i)
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

