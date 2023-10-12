
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
from .dim2_utils import dimN_dim2_inds,dimN_dim2_dists,dim2_dimN_dists,dim2_dimN_inds

def init(K,dim=1,anchor=False,descending=True,unqiue=False):
    def wrap(dists,inds):
        return run(dists,inds,K,dim)
    return wrap

def apply_topk(tensor,order,dim):


    # -- squash --
    shape = list(tensor.shape)
    tensor = dimN_dim2_dists(tensor,dim)[0]
    order = dimN_dim2_dists(order,dim)[0]

    # -- exec --
    tensor_k = th.gather(tensor,1,order)

    # -- shape back --
    shape[dim] = order.shape[1]
    tensor = dim2_dimN_dists(tensor_k,shape,dim)
    return tensor

def apply_topk_3d(tensor,order,dim):

    # -- squash --
    shape = list(tensor.shape)
    tensor = dimN_dim2_inds(tensor,dim)
    order = dimN_dim2_dists(order,dim)

    # -- allocate --
    device = tensor.device
    dtype = tensor.dtype
    Q,K,D = tensor.shape
    tensor_k = th.zeros((Q,K,D),device=device,dtype=dtype)

    # -- exec --
    for i in range(tensor.shape[-1]):
        tensor_k[:,:,i] = th.gather(tensor[:,:,i],1,order)

    # -- shape back --
    shape[dim] = K
    tensor = dim2_dimN_inds(tensor_k,shape,dim)
    return tensor

def run(dists,inds,k,dim=1,anchor=False,descending=True,
        unique=False,return_order=False):
    """

    Wrap the topk menu so the input to top-k is always square

    """
    # -- no run if k <= 0 --
    if not(k > 0): return dists,inds,None
    if unique: assert return_order == False

    # -- get squares --
    dists,inds,dshape,ishape = dimN_dim2(dists,inds,dim)

    # -- run top-k --
    dists,inds,order = topk_menu(dists,inds,k,anchor,descending,unique)

    # -- return squares --
    k = inds.shape[1]
    dists,inds = dim2_dimN(dists,inds,dshape,ishape,dim,k)
    if return_order:
        order = dim2_dimN_dists(order,dshape,dim)
        return dists,inds,order
    else:
        return dists,inds

def topk_menu(dists,inds,k,anchor=False,descending=True,
              unique=False):
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

    # -- sort non-anchor --
    _dists = dists[:,1:]
    _inds = inds[:,1:]
    k_s = _dists.shape[1] if unique else k-1
    _dists,_inds,_order = standard_topk(_dists,_inds,k_s,descending)

    # -- combine with anchor --
    dists = th.cat([dists0,_dists],1)
    inds = th.cat([inds0,_inds],1)
    order = th.cat([th.zeros_like(dists0).int(),_order+1],1)

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

    return dists,inds,order


def unique_topk(dists,inds,K,descending=False):


    # -- sort by dists --
    args = th.argsort(dists,dim=1,descending=descending)
    dists = th.gather(dists,1,args)
    for i in range(inds.shape[-1]):
        inds[...,i] = th.gather(inds[...,i],1,args)

    # -- run --
    dists_topk,inds_topk = unique_select(dists,inds,K,descending)

    # -- return --
    return dists_topk,inds_topk,None

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
    order_k = th.argsort(dists,dim=1,descending=descending)[:,:K]
    K = order_k.shape[1]

    # -- topk dists --
    dists_k = th.gather(dists,1,order_k)

    # -- topk inds --
    inds_k = th.zeros((Q,K,d2or3),device=inds.device,dtype=inds.dtype)
    for i in range(inds.shape[-1]):
        inds_k[:,:,i] = th.gather(inds[:,:,i],1,order_k)

    return dists_k,inds_k,order_k


# def run_refine(dists,inds,ksel,K,descending,anchor_self=False):
#     if anchor_self:
#         dists0 = dists[...,[0]]
#         inds0 = inds[...,[0],:]
#         ksel0 = ksel[...,[0]]
#         dists_k,inds_k,ksel_k = topk_each_impl(dists[...,1:],inds[...,1:,:],
#                                                ksel[...,1:],K-1,descending)
#         dists = th.stack([dists0,dists_k],-1)
#         inds = th.stack([inds0,inds_k],-2)
#         ksel = th.stack([ksel0,ksel_k],-2)
#     else:
#         dists,inds,ksel = topk_each_impl(dists,inds,K-1,descending)
#     return dists,inds,ksel

# def topk_each_impl(dists,inds,ksel,K,descending):

#     # -- reshape exh --
#     Q,S = dists.shape
#     d2or3 = inds.shape[-1]

#     # -- order --
#     order_k = th.argsort(dists,dim=1,descending=descending)[:,:K]
#     K = order_k.shape[1]

#     # -- topk dists --
#     dists_k = th.gather(dists,1,order_k)

#     # -- topk inds --
#     inds_k = th.zeros((Q,K,d2or3),device=inds.device,dtype=inds.dtype)
#     for i in range(inds.shape[-1]):
#         inds_k[:,:,i] = th.gather(inds[:,:,i],1,order_k)

#     # -- topk ksel
#     ksel_k = th.gather(ksel,1,order_k)

#     return dists_k,inds_k,ksel_k



def run_each(dists,inds,K,descending,anchor_self=False):
    if K <= 0:
        return dists,inds

    if anchor_self:
        dists0 = dists[...,[0]]
        inds0 = inds[...,[0],:]
        if K > 1:
            dists_k,inds_k = topk_each_impl(dists[...,1:],inds[...,1:,:],K-1,descending)
            dists = th.cat([dists0,dists_k],-1)
            inds = th.cat([inds0,inds_k],-2)
        else:
            dists = dists0
            inds = inds0
    else:
        dists,inds = topk_each_impl(dists,inds,K,descending)
    return dists,inds

def topk_each_impl(dists,inds,K,descending):

    # -- reshape --
    shape = list(dists.shape)
    G,S,d2or3= inds.shape[-3:]
    dists = dists.view(-1,S)
    inds = inds.view(-1,S,d2or3)
    Q = inds.shape[0]

    # -- order --
    order_k = th.argsort(dists,dim=1,descending=descending)[:,:K]

    # -- topk dists --
    dists_k = th.gather(dists,1,order_k)

    # -- topk inds --
    inds_k = th.zeros((Q,K,d2or3),device=inds.device,dtype=inds.dtype)
    for i in range(inds.shape[-1]):
        inds_k[:,:,i] = th.gather(inds[:,:,i],1,order_k)

    # -- view --
    shape[-1] = K
    dists_k = dists_k.view(shape)
    inds_k = inds_k.view(shape+[d2or3,])

    return dists_k,inds_k



