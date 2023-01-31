
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
    inds,ishape = dimN_dim2_inds(inds,dim)

    # -- K shape --
    K = K if K > 0 else inds.shape[3]

    # -- interpolate (K) neighbors --
    dnls_cuda.jitter_unique_inds(inds,K,H,W)

    # -- reshape --
    inds = dim2_dimN_inds(inds,ishape,dim)

    return inds
