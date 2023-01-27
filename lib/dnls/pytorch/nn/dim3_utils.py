
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
    dshape = dists.shape
    ishape = inds.shape
    dists = run_permute(dists)
    inds = run_ipermute(inds)
    return dists,inds,dshape,ishape

def dim3_dimN(dists,inds,dshape,ishape):
    dists = dists.reshape(dshape)
    inds = inds.reshape(ishape)
    return dists,inds

def run_permute(tensor):
    Q,K = tensor.shape[-2:]
    return tensor.reshape(-1,Q,K).contiguous()

def inv_permute(tensor,shape):
    return tensor.reshape(shape)

def run_ipermute(tensor):
    Q,K,_ = tensor.shape[-3:]
    return tensor.reshape(-1,Q,K,3).contiguous()

def inv_ipermute(tensor,shape):
    return tensor.reshape(shape)
