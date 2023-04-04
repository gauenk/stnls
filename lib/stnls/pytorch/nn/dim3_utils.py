
"""

Convert (N-dim <-> 3-dim)
with (....,Q,K) -> (B,Q,K)

This standardizes the input into other kernels

"""

# -- shaping imports --
import torch as th
from einops import rearrange,repeat

__all__ = ["dimN_dim3","dim3_dimN"]

def dimN_dim3(dists,inds):
    dists,dshape = dimN_dim3_dists(dists)
    inds,ishape = dimN_dim3_inds(inds)
    return dists,inds,dshape,ishape

def dim3_dimN(dists,inds,dshape,ishape):
    dists = dists.reshape(dshape)
    inds = inds.reshape(ishape)
    return dists,inds

def dimN_dim3_dists(tensor):
    shape = tensor.shape
    Q,K = tensor.shape[-2:]
    return tensor.reshape(-1,Q,K).contiguous(),shape

def dimN_dim3_inds(tensor):
    shape = tensor.shape
    Q,K,_ = tensor.shape[-3:]
    return tensor.reshape(-1,Q,K,3).contiguous(),shape
