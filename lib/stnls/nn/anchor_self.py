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

def run(dists,inds,stride0,nH,nW,qstart=0):

    # -- view --
    # print(dists.shape)
    B,HD,Q,Ks,ws,ws = dists.shape
    d2or3 = inds.shape[-1]
    dshape,ishape = list(dists.shape),list(inds.shape)
    dists = dists.view(B*HD,Q,Ks*ws*ws)
    inds = inds.view(B*HD,Q,Ks*ws*ws,d2or3)

    # -- [patchwork] --
    if d2or3 == 2:
        inds = th.cat([th.zeros_like(inds[...,[0]]),inds],-1)

    # -- allocate --
    order = th.zeros_like(dists[...,0]).int()

    # -- run --
    stnls_cuda.anchor_self(dists,inds,order,stride0,nH,nW)

    # -- [patchwork] --
    if d2or3 == 2:
        inds = inds[...,1:].contiguous()

    # -- return --
    dists = dists.reshape(dshape)
    inds = inds.reshape(ishape)
    order = order.reshape(dshape[:-3])

    return order

def run_refine(dists,inds,flows,stride0,qH,qW,kH,kW):

    # -- view --
    HD_f = flows.shape[1]
    if dists.ndim == 8:
        B,HD,T,nH,nW,Ks,ws,ws = dists.shape
        dists = dists.view(B,HD,T*nH*nW,Ks,ws*ws)
        inds = inds.view(B,HD,T*nH*nW,Ks,ws*ws,3)
        flows = flows.view(B,HD_f,T*nH*nW,Ks,3)
    assert inds.shape[-1] == 3,"Index must be size 3."
    # elif dists.ndim == 6:
    #     B,HD,T,nH,nW,Ks,ws,ws = dists.shape
    #     dists = dists.view(B,HD,T*nH*nW,Ks,ws*ws)
    #     inds = inds.view(B,HD,T*nH*nW,Ks,ws*ws,3)
    #     flows = flows.view(B,HD_f,T*nH*nW,Ks,3)
    # print("dists.shape,inds.shape,flows.shape: ",dists.shape,inds.shape,flows.shape)

    # -- run --
    stnls_cuda.anchor_self_refine(dists,inds,flows,stride0,qH,qW,kH,kW)

def run_time(dists,inds,flows,wt,stride0,qH,qW,kH,kW):

    # -- view --
    B,HD,Q,W_t,ws,ws = dists.shape
    d2or3 = inds.shape[-1]
    dists = dists.view(B,HD,Q,W_t,ws*ws)
    inds = inds.view(B,HD,Q,W_t,ws*ws,d2or3)
    assert d2or3 == 3,"Index must be size 3."

    # -- run --
    stnls_cuda.anchor_self_time(dists,inds,flows,wt,stride0,qH,qW,kH,kW)

def run_paired(dists,inds,flows,stride0,qH,qW,kH,kW):

    # -- view --
    B,HD,Q,G,ws,ws = dists.shape
    d2or3 = inds.shape[-1]
    dists = dists.view(B,HD,Q,G,ws*ws)
    inds = inds.view(B,HD,Q,G,ws*ws,d2or3)
    assert d2or3 == 2,"Index must be size 2."
    HD_f,nH,nW = flows.shape[1:4]
    msg = "Must match "+str(flows.shape)+" "+str((B,HD_f,nH,nW,G,2))
    assert flows.shape == (B,HD_f,nH,nW,G,2),msg
    # flows.shape = B,HD,nH,nW,G,two

    # -- run --
    stnls_cuda.anchor_self_paired(dists,inds,flows,stride0,qH,qW,kH,kW)


