"""

Compute the top-k pwd using not using a specified cuda kernel

"""

import torch as th
from dnls.pytorch.tile_k import unfold_k
from einops import rearrange

def run(vid,inds0,inds1,ps,pt=1,dilation=1,
        reflect_bounds=True,use_adj=False,
        off_H0=0,off_W0=0,off_H1=0,off_W1=0):

    # -- reshape --
    adj = ps//2 if use_adj else 0
    B,HD,T,C,H,W = vid.shape
    B,HD,Q,K,_ = inds0.shape
    vid = rearrange(vid,'b hd t c h w -> (b hd) t c h w')
    inds0 = rearrange(inds0,'b hd q k tr -> (b hd) q k tr')
    inds1 = rearrange(inds1,'b hd q k tr -> (b hd) q k tr')

    # -- unfold --
    patches0 = unfold_k(vid,inds0,ps,pt,dilation,
                        adj=adj,reflect_bounds=reflect_bounds)
    patches1 = unfold_k(vid,inds1,ps,pt,dilation,
                        adj=adj,reflect_bounds=reflect_bounds)

    # -- prepare --
    # BHD,Q,T,C,H,W = patches0.shape
    shape = "bhd q k t c h w -> (bhd q) k (t c h w)"
    patches0 = rearrange(patches0,shape)
    patches1 = rearrange(patches1,shape)

    # -- compute --
    pwd = th.cdist(patches0,patches1,p=2.0)

    # -- reshape --
    shape = "(b hd q) k0 k1 -> b hd q (k0 k1)"
    pwd = rearrange(pwd,shape,b=B,q=Q)
    k0,k1 = th.tril_indices(K,K,-1)
    kinds = k1*K+k0
    pwd = pwd[...,kinds].contiguous()
    # pwd = th.sort(pwd,-1)[0].contiguous()

    return pwd

