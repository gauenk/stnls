"""

Create Similar Images (fwd,bwd)


"""

# -- python --
import torch as th
import numpy as np
from einops import rearrange

# -- cpp cuda kernel --
import dnls_cuda

# -- package --
import dnls


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Pytorch Module
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class SimFwdBwd(th.nn.Module):

    def __init__(self, warp_ps, ws, ps, k, stride0=4, dist_type="l2", stride1=1,
                 dilation=1, pt=1, reflect_bounds=True, use_adj=False,
                 full_ws=True, rbwd=True, nbwd=1, exact=False, queries_per_thread=4,
                 neigh_per_thread=4, channel_groups=-1):
        super().__init__()


        # -- warp params --
        self.warp_ps = warp_ps
        self.k = k
        self.ws = ws
        self.stride0 = stride0
        self.dist_type = dist_type

        # -- search function --
        wt = 1
        nls_k = -1
        nls = dnls.search.NonLocalSearch
        self.search = nls(ws, wt, ps, nls_k, nheads=1,
                          dist_type=dist_type, stride0=stride0, stride1=stride1,
                          dilation=dilation, pt=pt, reflect_bounds=reflect_bounds,
                          full_ws=full_ws, anchor_self=False, remove_self=False,
                          use_adj=use_adj, rbwd=rbwd, nbwd=nbwd, exact=exact,
                          queries_per_thread=queries_per_thread,
                          neigh_per_thread=neigh_per_thread,
                          channel_groups=channel_groups)

    def forward(self,vid, fflow, bflow, batchsize=-1):

        # -- fix flow dims if needed --
        fflow,bflow = optional_flow_pad(vid,fflow,bflow)

        # -- run search --
        dists,inds = self.search(vid,vid,fflow,bflow)

        # -- top-K across time --
        descending = self.dist_type == "prod"
        dists,inds = dnls.nn.topk_time(dists,inds,self.k,self.ws,dim=3,
                                       descending=descending,anchor=False)

        # -- check --
        # # print(th.where(th.isinf(dists[:,:,:,1])))
        # args = th.where(th.isinf(dists[:,:,:,1]))
        # print(dists[:,:,args[2]])

        # -- get warped pairs --
        fwd,bwd = get_warps(vid,inds,self.warp_ps,self.k,self.stride0)

        return fwd,bwd

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Implementation Details
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def optional_flow_pad(vid,fflow,bflow):
    T = vid.shape[1]
    if fflow.shape[1] == T-1:
        zflow = th.zeros_like(fflow[:,[0]])
        fflow = th.cat([fflow,zflow],1)
        bflow = th.cat([zflow,bflow],1)
    return fflow,bflow

def get_warps(vid,inds,warp_ps,K,stride0):

    # -- unpack --
    T = vid.shape[1]
    B,HD,Q,_,_ = inds.shape
    Qt = Q//T
    vid = vid.contiguous()

    # -- pick across time --
    inds_ref = inds[:,:,:,[0]]
    inds_fwd = th.zeros((T,B,1,Qt,K,3),device=inds.device,dtype=inds.dtype)
    inds_bwd = th.zeros((T,B,1,Qt,K,3),device=inds.device,dtype=inds.dtype)
    for t in range(T):

        # -- fwd (t to t+1) --
        if t < (T-1):

            # -- get inds @ t+1 --
            args_t = th.where(inds_ref[...,0] == t+1)
            inds_t = inds[:,:,args_t[2]]

            # -- best match in t --
            args_k = th.where(inds_t[...,0] == t) # should be one each
            for i in range(3):
                inds_fwd[t][...,i] = inds_t[...,i][args_k].reshape(1,1,Qt,K)

        # -- bwd (t-1 to t) --
        if t > 0:

            # -- get inds @ t+1 --
            args_t = th.where(inds_ref[...,0] == t)
            inds_t = inds[:,:,args_t[2]]

            # -- best match in t --
            args_k = th.where(inds_t[...,0] == t-1) # should be one each
            for i in range(3):
                inds_bwd[t-1][...,i] = inds_t[...,i][args_k].reshape(1,1,Qt,K)

    # -- reshape --
    inds_fwd = rearrange(inds_fwd,'t b hd q k tr -> b hd (t q) k tr')
    inds_bwd = rearrange(inds_bwd,'t b hd q k tr -> b hd (t q) k tr')
    fwd = warp_video_inds(vid,inds_fwd,warp_ps,stride0)
    bwd = warp_video_inds(vid,inds_bwd,warp_ps,stride0)
    fwd[:,-1] = 0
    bwd[:,0] = 0

    return fwd,bwd

def warp_video_inds(vid,inds,ps,stride0):

    # -- patches --
    UnfoldK = dnls.UnfoldK(ps,adj=0)
    patches = UnfoldK(vid,inds[:,0]) # nheads == 1
    K = patches.shape[2]
    # assert K == 1
    patches = rearrange(patches,'b q k pt c ph pw -> (b k) q 1 pt c ph pw')

    # -- fold --
    B = vid.shape[0]
    vshape = [B*K,] + list(vid.shape[1:])
    fold = dnls.iFoldz(vshape,None,stride=stride0)
    fold(patches)
    warp = fold.vid / fold.zvid

    # -- stack channels --
    warp = rearrange(warp,'(b k) t c h w -> b t (k c) h w',k=K)

    return warp

