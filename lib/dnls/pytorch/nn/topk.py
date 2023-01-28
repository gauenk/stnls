
# -- import misc --
import torch as th
from einops import rearrange,repeat
import torch.nn.functional as nnf

# -- our package --
import dnls_cuda

# -- local --
# from .anchor_self import run as anchor_self

# -- import shape info --
# from .shape_utils import dimN_dim2,dim2_dimN
from .dim2_utils import dimN_dim2,dim2_dimN

def init(K,dim=1,anchor=False,descending=True,unqiue=False):
    def wrap(dists,inds):
        return run(dists,inds,K,dim)
    return wrap

def run(dists,inds,k,dim=1,anchor=False,descending=True,unique=False):
    """

    Wrap the topk menu so the input to top-k is always square

    """
    # -- no run if k <= 0 --
    if not(k > 0): return dists,inds

    # -- get squares --
    dists,inds,dshape,ishape = dimN_dim2(dists,inds,dim)

    # -- run top-k --
    dists,inds = topk_menu(dists,inds,k,anchor,descending,unique)

    # -- return squares --
    dists,inds = dim2_dimN(dists,inds,dshape,ishape,dim,k)
    return dists,inds

def topk_menu(dists,inds,k,anchor=False,descending=True,unique=False):
    """

    Select which topk to run

    """
    if anchor:
        return anchored_topk(dists,inds,k,descending,unique)
    elif unique:
        return unique_topk(dists,inds,k,descending)
    else:
        return standard_topk(dists,inds,k,descending)

def anchored_topk(dists,inds,k,descending,unique):

    # -- unpack first --
    dists0 = dists[:,[0]]
    inds0 = inds[:,[0]]

    # -- unpack dists --
    _dists = dists[:,1:]
    _inds = inds[:,1:]

    # -- find top-(k-1) --
    dists_k,inds_k = topk_menu(_dists,_inds,k-1,anchor=False,
                               descending=descending,unique=unique)

    # -- combine with anchor --
    dists = th.cat([dists0,dists_k],1)
    inds = th.cat([inds0,inds_k],1)

    return dists,inds


def unique_topk(dists,inds,K,descending=False,unique=True):

    # -- allocate --
    device = dists.device
    Q,S = dists.shape
    dists_topk = th.inf*th.ones((Q,K),device=device,dtype=dists.dtype)
    inds_topk = -th.ones((Q,K,3),device=device,dtype=inds.dtype)

    # -- sort by dists --
    args = th.argsort(dists,dim=1,descending=descending)
    dists = th.gather(dists,1,args)
    for i in range(3):
        inds[...,i] = th.gather(inds[...,i],1,args)

    # -- run --
    if unique:
        dnls_cuda.unique_topk(dists,inds,dists_topk,inds_topk,K)

    # -- return --
    return dists_topk,inds_topk

def standard_topk(dists,inds,K,descending):

    # -- reshape exh --
    Q,S = dists.shape

    # -- order --
    order = th.argsort(dists,dim=1,descending=descending)[:,:K]

    # -- topk dists --
    dists_k = th.gather(dists,1,order)

    # -- topk inds --
    inds_k = th.zeros((Q,K,3),device=inds.device,dtype=inds.dtype)
    for i in range(inds.shape[-1]):
        inds_k[:,:,i] = th.gather(inds[:,:,i],1,order)

    return dists_k,inds_k


