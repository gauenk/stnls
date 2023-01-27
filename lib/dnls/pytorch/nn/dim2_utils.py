
# -- shaping imports --
import torch as th
from einops import rearrange,repeat

def dimN_dim2(dists,inds,dim):
    dshape = list(dists.shape)
    ishape = list(inds.shape)
    dists = run_permute(dists,dim)
    inds = run_ipermute(inds,dim)
    return dists,inds,dshape,ishape

def dim2_dimN(dists,inds,dshape,ishape,dim,K):
    dshape[dim] = K
    ishape[dim] = K
    dists = inv_permute(dists,dshape,dim)
    inds = inv_ipermute(inds,ishape,dim)
    return dists,inds

def run_permute(tensor,dim):
    """

    paired with "inv_permute"
    Transforms tensor: (a1,a2,...,aN) -> (a1 x a2 x ... x aN, aX)

    """
    D = tensor.shape[dim]
    tensor = tensor.transpose(0,dim).reshape(D,-1).T
    return tensor.contiguous()

def inv_permute(tensor,shape,dim):
    """

    paired with "run_permute"
    Transforms tensor: (a1 x a2 x ... x aN, aX) -> (a1,a2,...,aN)

    """

    # -- swap --
    shape = list(shape)
    tmp = shape[dim]
    shape[dim] = shape[0]
    shape[0] = tmp

    # -- reshape --
    return tensor.T.reshape(shape).transpose(dim,0)


def run_ipermute(tensor,dim):
    """

    paired with "run_ipermute"
    Transforms tensor: (a1,a2,...,aN) -> (a1 x a2 x ... x aN, aX)

    """
    D = tensor.shape[dim]
    tensor = tensor.transpose(0,dim).reshape(D,-1,3).transpose(0,1)
    return tensor.contiguous()

def inv_ipermute(tensor,shape,dim):
    """

    paired with "inv_ipermute"
    Transforms tensor: (a1 x a2 x ... x aN, aX, 3) -> (a1,a2,...,aN,3)

    """

    # -- swap --
    shape = list(shape)
    tmp = shape[dim]
    shape[dim] = shape[0]
    shape[0] = tmp

    # -- reshape --
    return tensor.transpose(1,0).reshape(shape).transpose(dim,0)
