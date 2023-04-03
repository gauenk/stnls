"""
Anchor the self-patch displace as the first index.

This is a nice ordering for many subsequent routines.

Using Pytorch functions such as "mask" consumes huge GPU Mem.

We can't just compute center of "wt,ws,ws" since our search
space is not always, nor should be, centered. This is really
only true at image boundaries... So silly.

"""

import torch as th
import dnls_cuda
from .dim3_utils import dimN_dim3,dim3_dimN

def run(dists,inds,stride0,H,W,qstart=0):
    dists,inds,dshape,ishape = dimN_dim3(dists,inds)
    dnls_cuda.anchor_self(dists,inds,qstart,stride0,H,W)
    dists,inds = dim3_dimN(dists,inds,dshape,ishape)
    return dists,inds
