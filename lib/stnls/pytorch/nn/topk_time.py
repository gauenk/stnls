"""

Compute TopK across the time window

"""

# -- import misc --
import torch as th
from einops import rearrange,repeat
import torch.nn.functional as nnf

# -- our package --
import stnls_cuda
import stnls

# -- local --
from .dim2_utils import dimN_dim2,dim2_dimN

# # -- share --
# from ..search.shared import manage_self

def run(dists,inds,k,ws,dim=1,anchor=True,
        descending=True,unique=False,include_self_time=False):

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      Reshape to Standard Dim
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- no run if k <= 0 --
    if not(k > 0): return dists,inds

    # -- get squares --
    dists,inds,dshape,ishape = dimN_dim2(dists,inds,dim)

    # -- run top-k --
    dists,inds = topk_menu(dists,inds,k,ws,anchor,descending,
                           unique,include_self_time)

    # -- return squares --
    k = inds.shape[1]
    dists,inds = dim2_dimN(dists,inds,dshape,ishape,dim,k)
    return dists,inds

def topk_menu(dists,inds,k,ws,anchor=False,
              descending=True,unique=False,include_self_time=False):
    """

    Select which topk to run

    """
    assert anchor == True
    assert unique == False
    # if anchor:
    #     return anchored_topk(dists,inds,k,ws,descending,unique)
    # elif unique:
    #     return unique_topk(dists,inds,k,descending)
    # else:
    #     return standard_topk(dists,inds,k,ws,descending)
    standard_topk_time(dists,inds,k,ws,descending,include_self_time)

# def anchored_topk(dists,inds,k,ws,descending,unique):

#     # -- unpack first --
#     dists0 = dists[:,[0]]
#     inds0 = inds[:,[0]]

#     # -- sort non-anchor --
#     _dists = dists[:,1:]
#     _inds = inds[:,1:]
#     k_s = _dists.shape[1] if unique else k-1
#     _dists,_inds = standard_topk(_dists,_inds,k_s,ws,descending)

#     # -- combine with anchor --
#     dists = th.cat([dists0,_dists],1)
#     inds = th.cat([inds0,_inds],1)

#     # -- run --
#     if unique:
#         dists,inds = unique_select(dists,inds,k,descending)

#     return dists,inds


def standard_topk(dists,inds,K,ws,descending,include_self_time):

    # -- reshape round 2 --
    shape_str = 'b (sT wsH wsW) -> b (wsH wsW) sT'
    dists = rearrange(dists,shape_str,wsH=ws,wsW=ws)
    shape_str = 'b (sT wsH wsW) tr -> b (wsH wsW) sT tr'
    inds = rearrange(inds,shape_str,wsH=ws,wsW=ws)

    # -- reshape exh --
    Q,sW,sT = dists.shape

    # -- order --
    order = th.argsort(dists,dim=1,descending=descending)#[:,:K]
    oK = order.shape[1]
    print(order)
    exit()
    zidx = th.where(order == 0)
    print(order.shape)
    order = th.cat([order[zidx],order[:zidx],order[zidx+1:]],0) # anchor self

    # -- topk dists --
    dists_k = th.gather(dists,1,order)

    # -- topk inds --
    inds_k = th.zeros((Q,oK,sT,3),device=inds.device,dtype=inds.dtype)
    for i in range(inds.shape[-1]):
        inds_k[:,:,:,i] = th.gather(inds[:,:,:,i],1,order)

    # -- remove remainder of sT == 0 --
    # dists_k = th.cat([dists_k[:,:2,[t]].reshape(Q,2) for t in range(sT)],1)
    # inds_k = th.cat([inds_k[:,:2,[t],:].reshape(Q,2,3) for t in range(sT)],1)
    num = K
    t0 = num if include_self_time else 1
    dists_k = th.cat(
        [dists_k[:,:num,[0]].reshape(Q,1),] +
        [dists_k[:,:num,[t]].reshape(Q,num) for t in range(1,sT)],1)
    inds_k = th.cat(
        [inds_k[:,:num,[0]].reshape(Q,1,3),] +
        [inds_k[:,:num,[t],:].reshape(Q,num,3) for t in range(1,sT)],1)


    # dists_k = th.cat([dists_k[:,[0],[0]].reshape(Q,1),
    #                   dists_k[:,1:,1:].reshape(Q,-1)],1)
    # inds_k = th.cat([inds_k[:,[0],[0],:].reshape(Q,1,3),
    #                  inds_k[:,1:,1:,:].reshape(Q,-1,3)],1)

    # # -- restore old shape --
    # dists_k = rearrange(dists_k,'q sT sW -> q (sT sW)')
    # inds_k = rearrange(inds_k,'q sT sW tr -> q (sT sW) tr')

    return dists_k,inds_k

def unique_topk(dists,inds,K,descending=False):

    raise NotImplementedError("")

    # -- sort by dists --
    args = th.argsort(dists,dim=1,descending=descending)
    dists = th.gather(dists,1,args)
    for i in range(3):
        inds[...,i] = th.gather(inds[...,i],1,args)

    # -- run --
    dists_topk,inds_topk = unique_select(dists,inds,K,descending)

    # -- return --
    return dists_topk,inds_topk

def unique_select(dists,inds,K,descending):
    raise NotImplementedError("")
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

    # -- allocate --
    dists_topk = th.zeros((Q,K),device=device,dtype=dtype)
    dists_topk[...] = -th.inf if descending else th.inf
    inds_topk = th.zeros((Q,K,3),device=device,dtype=itype)
    inds_topk[...] = -1
    return dists_topk,inds_topk
