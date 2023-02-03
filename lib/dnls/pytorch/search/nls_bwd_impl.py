
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
    # assert not(th.any(inds==-1).item()),"No -1 indices"

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
    qshift = 0 # no batching backward.

    # -- allow for repeated exec --
    # torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    # torch::Tensor vid0, torch::Tensor vid1,
    # torch::Tensor grad_dists, torch::Tensor inds,
    # int q_shift, int stride0, int nH0, int nW0,
    # int ps, int pt, int dilation, bool reflect_bounds,
    # bool use_adj, int off_H0, int off_W0, int off_H1, int off_W1,
    # bool use_rand, bool exact, int dist_type,
    # int channel_groups, int neigh_per_thread, int queries_per_thread) {

    bwd_fxn = dnls_cuda.non_local_search_backward
    if ctx.nbwd == 1:
        bwd_fxn(grad_vid0,grad_vid1,vid0,vid1,
                grad_dists,inds,qshift,ctx.stride0,nH0,nW0,
                ctx.ps,ctx.pt,ctx.dil,ctx.reflect_bounds,ctx.use_adj,
                ctx.off_H0, ctx.off_W0,ctx.off_H1, ctx.off_W1,
                ctx.rbwd,ctx.exact,ctx.dist_type_i,
                ctx.channel_groups,ctx.neigh_per_thread,ctx.queries_per_thread)
    else:
        for _ in range(ctx.nbwd):
            grad_vid0_i = allocate_vid(vid_shape,grad_dists.device)
            grad_vid1_i = allocate_vid(vid_shape,grad_dists.device)
            bwd_fxn(grad_vid0_i,grad_vid1_i,vid0,vid1,
                    grad_dists,inds,qshift,ctx.stride0,nH0,nW0,
                    ctx.ps,ctx.pt,ctx.dil,ctx.reflect_bounds,ctx.use_adj,
                    ctx.off_H0, ctx.off_W0,ctx.off_H1, ctx.off_W1,
                    ctx.rbwd,ctx.exact,ctx.dist_type_i,
                    ctx.channel_groups,ctx.neigh_per_thread,ctx.queries_per_thread)
            grad_vid0 += grad_vid0_i
            grad_vid1 += grad_vid1_i
        grad_vid0 /= ctx.nbwd
        grad_vid1 /= ctx.nbwd

    # -- finalize shape --
    grad_vid0 = rearrange(grad_vid0,'B H t c h w -> B t (H c) h w')
    grad_vid1 = rearrange(grad_vid1,'B H t c h w -> B t (H c) h w')

    return grad_vid0,grad_vid1
