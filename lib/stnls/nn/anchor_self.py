"""
Anchor the self-patch displace as the first index.

This is a nice ordering for many subsequent routines.

Using Pytorch functions such as "mask" consumes huge GPU Mem.

We can't just compute center of "wt,ws,ws" since our search
space is not always, nor should be, centered. This is really
only true at image boundaries... So silly.

"""

import torch as th
import stnls_cuda
from .dim3_utils import dimN_dim3,dim3_dimN

def run(dists,inds,stride0,H,W,qstart=0):

    # -- view --
    # print(dists.shape)
    B,HD,Q,Ks,ws,ws = dists.shape
    dshape,ishape = list(dists.shape),list(inds.shape)
    dists = dists.view(B*HD,Q,Ks*ws*ws)
    inds = inds.view(B*HD,Q,Ks*ws*ws,3)

    # -- allocate --
    order = th.zeros_like(dists[...,0]).int()

    # -- run --
    stnls_cuda.anchor_self(dists,inds,order,stride0,H,W)

    # -- return --
    dists = dists.reshape(dshape)
    inds = inds.reshape(ishape)
    order = order.reshape(dshape[:-3])

    return order

def run_refine(dists,inds,flows,stride0,H,W):

    # -- view --
    B,HD,T,nH,nW,Ks,ws,ws = dists.shape
    dists = dists.view(B,HD,T*nH*nW,Ks,ws*ws)
    inds = inds.view(B,HD,T*nH*nW,Ks,ws*ws,3)
    HD_f = flows.shape[1]
    flows = flows.view(B,HD_f,T*nH*nW,Ks,3)

    # -- run --
    stnls_cuda.anchor_self_refine(dists,inds,flows,stride0,H,W)

def run_time(dists,inds,flows,wt,stride0,H,W):

    # -- view --
    B,HD,Q,W_t,ws,ws = dists.shape
    d2or3 = inds.shape[-1]
    dists = dists.view(B,HD,Q,W_t,ws*ws)
    inds = inds.view(B,HD,Q,W_t,ws*ws,d2or3)

    # -- run --
    stnls_cuda.anchor_self_time(dists,inds,flows,wt,stride0,H,W)

