
# -- shaping imports --
import torch as th
from einops import rearrange,repeat

def dimN_dim2(dists,inds,dim):
    dists,dshape = dimN_dim2_dists(dists,dim)
    inds,ishape = dimN_dim2_inds(inds,dim)
    return dists,inds,dshape,ishape

def dim2_dimN(dists,inds,dshape,ishape,dim,K):
    dshape[dim] = K
    ishape[dim] = K
    dists = dim2_dimN_dists(dists,dshape,dim)
    inds = dim2_dimN_inds(inds,ishape,dim)
    return dists,inds

def dimN_dim2_dists(tensor,dim):
    """

    paired with "dim2_dimN_dists"
    Transforms tensor: (a1,a2,...,aN) -> (a1 x a2 x ... x aN, aX)

    """
    shape = tensor.shape
    D = tensor.shape[dim]
    tensor = tensor.transpose(0,dim).reshape(D,-1).T
    return tensor.contiguous(),list(shape)

def dim2_dimN_dists(tensor,shape,dim):
    """

    paired with "dimN_dim2_dists"
    Transforms tensor: (a1 x a2 x ... x aN, aX) -> (a1,a2,...,aN)

    """

    # -- swap --
    shape = list(shape)
    tmp = shape[dim]
    shape[dim] = shape[0]
    shape[0] = tmp

    # -- reshape --
    return tensor.T.reshape(shape).transpose(dim,0)



def dimN_dim2_inds(tensor,dim):
    """

    paired with "dimN_dim2_inds"
    Transforms tensor: (a1,a2,...,aN) -> (a1 x a2 x ... x aN, aX)

    """
    shape = tensor.shape
    D = tensor.shape[dim]
    tensor = tensor.transpose(0,dim).reshape(D,-1,3).transpose(0,1)
    return tensor.contiguous(),list(shape)

def dim2_dimN_inds(tensor,shape,dim):
    """

    paired with "dim2_dimN_inds"
    Transforms tensor: (a1 x a2 x ... x aN, aX, 3) -> (a1,a2,...,aN,3)

    """

    # -- swap --
    shape = list(shape)
    tmp = shape[dim]
    shape[dim] = shape[0]
    shape[0] = tmp

    # -- reshape --
    return tensor.transpose(1,0).reshape(shape).transpose(dim,0)
