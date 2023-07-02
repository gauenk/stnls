
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
from .shared import manage_self

def nls_backward(ctx, grad_dists, grad_inds_is_none):

    # -- populate names --
    inds,vid0,vid1 = ctx.saved_tensors
    # print("inds.shape: ",inds.shape,vid0.shape,vid1.shape)
    # assert not(th.any(inds==-1).item()),"No -1 indices"
    # print(grad_dists.shape,inds.shape,ctx.vid_shape)

    # -- allocate grads --
    grad_vid0 = allocate_vid(ctx.vid_shape,grad_dists.device)
    grad_vid1 = allocate_vid(ctx.vid_shape,grad_dists.device)

    # -- restrict to k_agg --
    if ctx.k_agg > 0:
        grad_dists = grad_dists[...,:ctx.k_agg]
        inds = inds[...,:ctx.k_agg,:]

    # -- ensure contiguous --
    grad_dists = grad_dists.contiguous()
    inds = inds.contiguous()

    # -- derived shapes --
    H,W = ctx.vid_shape[-2:]
    nH0 = (H-1)//ctx.stride0+1
    nW0 = (W-1)//ctx.stride0+1
    qshift = 0 # no batching backward.

    # -- debug --
    # vid0[...] = 2.
    # vid1[...] = 1.
    # grad_dists[...] = 1
    # print(ctx.reflect_bounds,ctx.use_adj)
    # ctx.reflect_bounds = True
    # ctx.reflect_bounds = False
    # ctx.use_adj = True
    # # grad_vid0[...] = 1
    # # grad_vid1[...] = 1

    # -- allow for repeated exec --
    bwd_fxn = stnls_cuda.non_local_search_backward
    bwd_fxn(grad_vid0,grad_vid1,vid0,vid1,
            grad_dists,inds,qshift,ctx.stride0,nH0,nW0,
            ctx.ps,ctx.pt,ctx.dil,ctx.reflect_bounds,ctx.use_adj,
            ctx.off_H0, ctx.off_W0,ctx.off_H1, ctx.off_W1,ctx.dist_type_i)


    # -- finalize shape --
    grad_vid0 = rearrange(grad_vid0,'B H t c h w -> B t (H c) h w')
    grad_vid1 = rearrange(grad_vid1,'B H t c h w -> B t (H c) h w')

    # -- normz --
    # from torch.nn.functional import fold
    from torchvision.transforms.functional import center_crop
    from stnls import iFoldz

    nH1 = (H-1)//ctx.stride1+1
    nW1 = (W-1)//ctx.stride1+1
    # H,W = grad_vid0.shape[-2:]
    # B,HD,Q,K = grad_dists.shape
    # counts = th.ones((B,1,H,W),device=grad_dists)
    # print((1,ctx.ps*ctx.ps,nH1*nW1))

    if ctx.normalize_bwd:
        reflect_bounds = ctx.reflect_bounds
        dilation = ctx.dil
        # print(grad_vid1.shape)
        fold = stnls.iFoldz(grad_vid1[:1,:1,:1].shape,
                            stride=ctx.stride0,dilation=dilation,
                            reflect_bounds=reflect_bounds,
                            device=vid1.device,use_adj=False)
        counts = th.ones((1,nH0*nW0,1,1,1,ctx.ps,ctx.ps),device=grad_dists.device)
        counts,_ = fold(counts)

        # counts = th.ones((1,ctx.ps*ctx.ps,nH0*nW0),device=grad_dists.device)
        # Hp = (nH0 - 1) * ctx.stride0 + ctx.ps
        # Wp = (nW0 - 1) * ctx.stride0 + ctx.ps
        # counts = fold(counts, (Hp,Wp), [ctx.ps]*2, 1, [0,0], ctx.stride0)
        # counts = center_crop(counts,(H,W))
        # sH,sW = (Hp-H+1)//2,(Wp-W+1)//2
        # counts = counts[...,sH:sH+H,sW:sW+W]

        # print("[nls] grad_vid0:")
        # print(grad_vid0[0,0,0,:3,:3])
        # print(counts[0,0,:3,:3])
        # grad_vid0 /= counts[None,]
        grad_vid0 /= counts

        # fold = stnls.iFoldz(grad_vid1[:1,:1,:1].shape,
        #                     stride=ctx.stride1,dilation=dilation,
        #                     reflect_bounds=reflect_bounds,
        #                     device=vid1.device,use_adj=False)
        # counts = th.ones((1,nH1*nW1,1,1,1,ctx.ps,ctx.ps),device=grad_dists.device)
        # counts,_ = fold(counts)

        from torch.nn.functional import fold
        counts = th.ones((1,ctx.ps*ctx.ps,nH1*nW1),device=grad_dists.device)
        Hp = (nH1 - 1) * ctx.stride1 + ctx.ps
        Wp = (nW1 - 1) * ctx.stride1 + ctx.ps
        counts = fold(counts, (Hp,Wp), [ctx.ps]*2, 1, [0,0], ctx.stride1)
        sH,sW = (Hp-H+1)//2,(Wp-W+1)//2
        counts = counts[...,sH:sH+H,sW:sW+W]

        # counts = center_crop(counts,(H,W))
        # print(counts)
        # print("counts.shape: ",counts.shape)
        # print("[nls] grad_vid1:")
        # print(grad_vid1[0,0,0,:3,:3])
        # print("[nls] counts: ")
        # print(counts[0,0,:3,:3])
        # grad_vid1 /= counts[None,]
        grad_vid1 /= counts

    return grad_vid0,grad_vid1

def nls_quad_backward(ctx, grad_dists, grad_inds_is_none):

    # -- populate names --
    inds,vid0,vid1,deno0,deno1 = ctx.saved_tensors
    # print("inds.shape: ",inds.shape,vid0.shape,vid1.shape)
    # assert not(th.any(inds==-1).item()),"No -1 indices"
    # print(grad_dists.shape,inds.shape,ctx.vid_shape)

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
    bwd_fxn = stnls_cuda.quadref_backward
    if ctx.nbwd == 1:
        bwd_fxn(grad_vid0,grad_vid1,vid0,vid1,deno0,deno1,
                grad_dists,inds,qshift,ctx.stride0,nH0,nW0,
                ctx.ps,ctx.pt,ctx.dil,ctx.reflect_bounds,ctx.use_adj,
                ctx.off_H0, ctx.off_W0,ctx.off_H1, ctx.off_W1,
                ctx.rbwd,ctx.exact,ctx.dist_type_i,
                ctx.queries_per_thread,ctx.neigh_per_thread,ctx.channel_groups)
    else:
        for _ in range(ctx.nbwd):
            grad_vid0_i = allocate_vid(vid_shape,grad_dists.device)
            grad_vid1_i = allocate_vid(vid_shape,grad_dists.device)
            bwd_fxn(grad_vid0_i,grad_vid1_i,vid0,vid1,deno0,deno1,
                    grad_dists,inds,qshift,ctx.stride0,nH0,nW0,
                    ctx.ps,ctx.pt,ctx.dil,ctx.reflect_bounds,ctx.use_adj,
                    ctx.off_H0, ctx.off_W0,ctx.off_H1, ctx.off_W1,
                    ctx.rbwd,ctx.exact,ctx.dist_type_i,
                    ctx.queries_per_thread,ctx.neigh_per_thread,ctx.channel_groups)
            grad_vid0 += grad_vid0_i
            grad_vid1 += grad_vid1_i
        grad_vid0 /= ctx.nbwd
        grad_vid1 /= ctx.nbwd

    # -- finalize shape --
    grad_vid0 = rearrange(grad_vid0,'B H t c h w -> B t (H c) h w')
    grad_vid1 = rearrange(grad_vid1,'B H t c h w -> B t (H c) h w')

    return grad_vid0,grad_vid1

