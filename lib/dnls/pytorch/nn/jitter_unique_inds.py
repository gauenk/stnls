
"""

Get unique indices by jittering the
possibly-nonunique input indices

"""

import torch as th
from einops import rearrange,repeat
import torch.nn.functional as nnf
import dnls_cuda
from .dim2_utils import dimN_dim2_inds,dim2_dimN_inds

def run(inds,dim,K,H,W):

    # -- shape to 2dim --
    ishape = dimN_dim3_inds(inds,dim)

    # -- interpolate (K) neighbors --
    dnls_cuda.jitter_unique_inds(inds,K,H,W)

    # -- reshape --
    inds = dim3_dimN_inds(inds,ishape,dim)

    return inds
