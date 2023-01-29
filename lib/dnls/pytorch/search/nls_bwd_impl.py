
# -- python --
import torch as th
import numpy as np
from einops import rearrange

# -- cpp cuda kernel --
import dnls_cuda

# -- package --
import dnls

# -- local --
from .utils import shape_vids,allocate_pair,dist_type_select,allocate_vid
from .shared import manage_self

def nls_backward(ctx, grad_dists, grad_inds_is_none):

    # -- populate names --
    inds,vid0,vid1 = ctx.saved_tensors

    # -- allocate grads --
    grad_vid0 = allocate_vid(ctx.vid_shape,grad_dists.device)
    grad_vid1 = allocate_vid(ctx.vid_shape,grad_dists.device)

    # -- ensure contiguous --
    grad_dists = grad_dists.contiguous()
    inds = inds.contiguous()

    # -- derived shapes --
    H,W = ctx.vid_shape[-2:]
    nH0 = (H-1)//ctx.stride0+1
    nW0 = (W-1)//ctx.stride0+1

    # -- allow for repeated exec --
    bwd_fxn = dnls_cuda.non_local_search_backward
    if ctx.nbwd == 1:
        bwd_fxn(grad_vid0,grad_vid1,vid0,vid1,
                grad_dists,inds,ctx.qshift,ctx.stride0,nH0,nW0,
                ctx.off_H0, ctx.off_W0,ctx.off_H1, ctx.off_W1,
                ctx.ps,ctx.pt,ctx.dil,ctx.use_adj,
                ctx.reflect_bounds,ctx.rbwd,ctx.exact,ctx.dist_type_i)
    else:
        for _ in range(ctx.nbwd):
            grad_vid0_i = allocate_vid(vid_shape,grad_dists.device)
            grad_vid1_i = allocate_vid(vid_shape,grad_dists.device)
            bwd_fxn(grad_vid0_i,grad_vid1_i,vid0,vid1,
                    grad_dists,inds,ctx.qshift,ctx.stride0,nH0,nW0,
                    ctx.off_H0, ctx.off_W0,ctx.off_H1, ctx.off_W1,
                    ctx.ps,ctx.pt,ctx.dil,ctx.use_adj,
                    ctx.reflect_bounds,ctx.rbwd,ctx.exact,ctx.dist_type_i)
            grad_vid0 += grad_vid0_i
            grad_vid1 += grad_vid1_i
        grad_vid0 /= ctx.nbwd
        grad_vid1 /= ctx.nbwd

    # -- finalize shape --
    grad_vid0 = rearrange(grad_vid0,'B H t c h w -> B t (H c) h w')
    grad_vid1 = rearrange(grad_vid1,'B H t c h w -> B t (H c) h w')

    return grad_vid0,grad_vid1,None,None,None,None,\
        None,None,None,None,None,None,None,None,None,None,None,\
        None,None,None,None,None,None,None,None,None,None,None
