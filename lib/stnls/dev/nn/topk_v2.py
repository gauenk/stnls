
# -- import misc --
import torch as th
from einops import rearrange,repeat
import torch.nn.functional as nnf

# -- our package --
import stnls_cuda
import stnls

# -- local --
# from .anchor_self import run as anchor_self

# -- import shape info --
# from .shape_utils import dimN_dim2,dim2_dimN
from .dim2_utils import dimN_dim2,dim2_dimN

def run(dists,k,dim=1,anchor=False,descending=True):
    """
    Wrap the topk menu so the input to top-k is always square
    """
    # -- no run if k <= 0 --
    if not(k > 0): return None

    # -- get squares --
    dshape = dists.shape
    dimN_dim2_dists
    dists = dim2_dimN_dists(tensor,shape,dim)
    # dists,inds,dshape,ishape = dimN_dim2(dists,inds,dim)

    # -- run top-k --
    dists,inds = topk_menu(dists,inds,k,anchor,descending,unique,qinds)

    # -- return squares --
    k = inds.shape[1]
    dists,inds = dim2_dimN(dists,inds,dshape,ishape,dim,k)
    return dists,inds

def topk_menu(dists,inds,k,anchor=False,descending=True,
              unique=False,qinds=None):
    """

    Select which topk to run

    """
    if anchor:
        return anchored_topk(dists,inds,k,descending,unique,qinds)
    elif unique:
        return unique_topk(dists,inds,k,descending)
    else:
        return standard_topk(dists,inds,k,descending)

def anchored_topk(dists,inds,k,descending,unique,qinds):

    # -- unpack first --
    dists0 = dists[:,[0]]
    inds0 = inds[:,[0]]

    # -- sort non-anchor --
    _dists = dists[:,1:]
    _inds = inds[:,1:]
    k_s = _dists.shape[1] if unique else k-1
    _dists,_inds = standard_topk(_dists,_inds,k_s,descending)

    # -- combine with anchor --
    dists = th.cat([dists0,_dists],1)
    inds = th.cat([inds0,_inds],1)

    # -- check -1 --
    # dists_tmp = dists.clone()
    # inds_tmp = inds.clone()
    # assert not(th.any(inds[:,:k]==-1).item()),"[%s] No -1 indices" % __file__

    # -- run --
    if unique:
        dists,inds = unique_select(dists,inds,k,descending)

    # -- check dups -
    # dups,any_dup = stnls.testing.find_duplicate_inds(inds)
    # args = th.where(dups == True)
    # if len(args[0]) > 0:
    #     print(inds.shape,dups.shape)
    #     print(inds[args[0][0]])
    #     print(dists[args[0][0]])
    #     print(dups[args[0][0]])
    #     print(inds_tmp[args[0][0]])
    #     print(dists_tmp[args[0][0]])
    # assert not(any_dup)

    # -- info --
    # args = th.where(inds==-1)
    # if len(args[0]) > 0:
    #     print(qinds[args[0][0]])
    #     print(inds_tmp[args[0][0]])
    #     print(inds[args[0][0]])

    # -- check -1 --
    # assert not(th.any(inds==-1).item()),"[%s] No -1 indices" % __file__

    return dists,inds


def unique_topk(dists,inds,K,descending=False):


    # -- sort by dists --
    args = th.argsort(dists,dim=1,descending=descending)
    dists = th.gather(dists,1,args)
    for i in range(inds.shape[-1]):
        inds[...,i] = th.gather(inds[...,i],1,args)

    # -- run --
    dists_topk,inds_topk = unique_select(dists,inds,K,descending)

    # -- return --
    return dists_topk,inds_topk

def unique_select(dists,inds,K,descending):
    inds = inds.contiguous()
    dists_topk,inds_topk = allocate_topk(dists,inds,K,descending)
    stnls_cuda.unique_topk(dists,inds,dists_topk,inds_topk,K)
    return dists_topk,inds_topk

def allocate_topk(dists,inds,K,descending):

    # -- unpack --
    Q,S = dists.shape
    device = dists.device
    dtype = dists.dtype
    itype = inds.dtype
    d2or3 = inds.shape[-1]

    # -- allocate --
    dists_topk = th.zeros((Q,K),device=device,dtype=dtype)
    dists_topk[...] = -th.inf if descending else th.inf
    inds_topk = th.zeros((Q,K,d2or3),device=device,dtype=itype)
    inds_topk[...] = -1
    return dists_topk,inds_topk

def standard_topk(dists,inds,K,descending):

    # -- reshape exh --
    Q,S = dists.shape
    d2or3 = inds.shape[-1]

    # -- order --
    order = th.argsort(dists,dim=1,descending=descending)[:,:K]
    K = order.shape[1]

    # -- topk dists --
    dists_k = th.gather(dists,1,order)

    # -- topk inds --
    inds_k = th.zeros((Q,K,d2or3),device=inds.device,dtype=inds.dtype)
    for i in range(inds.shape[-1]):
        inds_k[:,:,i] = th.gather(inds[:,:,i],1,order)

    return dists_k,inds_k


