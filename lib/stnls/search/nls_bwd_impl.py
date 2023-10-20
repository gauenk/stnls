
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

def nls_backward(ctx, grad_dists, grad_inds):

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

    # -- backward pass with increasing complexity --
    if ctx.itype == "int":
        bwd_fxn = stnls_cuda.non_local_search_int_vid_backward
        bwd_fxn(grad_vid0,grad_vid1,
                vid0,vid1,grad_dists,inds,
                ctx.ps,ctx.pt,ctx.stride0,ctx.dil,
                ctx.reflect_bounds,patch_offset,ctx.dist_type_i)
    elif not(flows.requires_grad):
        bwd_fxn = stnls_cuda.non_local_search_bilin2d_vid_backward
        bwd_fxn(grad_vid0,grad_vid1,vid0,vid1,
                grad_dists,inds,
                ctx.wt,ctx.ps,ctx.pt,ctx.stride0,ctx.dil,
                ctx.reflect_bounds,patch_offset,ctx.dist_type_i)
    else:
        bwd_fxn = stnls_cuda.non_local_search_bilin2d_vidflows_backward
        bwd_fxn(grad_vid0,grad_vid1,grad_flows,
                vid0,vid1,flows,
                grad_dists,grad_inds,dists,inds,
                ctx.wt,ctx.ps,ctx.pt,ctx.stride0,ctx.dil,
                ctx.reflect_bounds,patch_offset,ctx.dist_type_i)

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
