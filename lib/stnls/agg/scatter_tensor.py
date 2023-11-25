"""

  Compute the destination indices for NonLocalScatter
  to remove the race condition of center
  and mitigate the race condition of the patch

 flows_k MUST be output from a grid search of size (2*wt+1,ws,ws)
 offset from flows. This is not valid for aribtrary values of flows_k

"""


# -- imports --
import torch as th
from einops import rearrange

# -- cpp cuda kernel --
import stnls_cuda

def run(tensor,flows_k,labels,stride0,stride1,H,W):

    # -- unpack shapes --
    B,HD,T,nH0,nW0,K = tensor.shape[:6]
    Q0 = T*nH0*nW0
    S = labels.max().int()+1
    tensor = tensor.reshape(B,HD,Q0,K,-1)
    M = tensor.shape[-1]
    nH1 = (H-1)//stride1+1
    nW1 = (W-1)//stride1+1
    Q1 = T*nH1*nW1

    # -- change type if needed --
    dtype = tensor.dtype
    if dtype in [th.int32,th.int64]:
        tensor = tensor.float()

    # -- prepare --
    shape = (B,HD,Q1,S,M)
    scatter_tensor = -th.inf*th.ones(shape,device=labels.device,dtype=tensor.dtype)
    stnls_cuda.scatter_tensor_forward(scatter_tensor,tensor,labels,flows_k,
                                      stride0,stride1,H,W)

    # -- adjust output type --
    if dtype in [th.int32,th.int64]:
        scatter_tensor = scatter_tensor.type(dtype)

    # -- squeeze single M --
    if M == 1:
        scatter_tensor = scatter_tensor[...,0]

    return scatter_tensor

# def scatter_flows(flows,labels,stride0,stride1):

#     # -- imports --
#     import stnlts.utils.misc import get_space_grid

#     # -- unpack --
#     B,HD,T,nH0,nW0,K,_ = flows.shape
#     nH1,nW1 = (nH0*stride0)//stride1,(nW0*stride0)//stride1

#     # -- get grids --
#     grid0 = get_space_grid(nH0,nW0,dtype=th.float,device="cuda")
#     grid0 = grid0[:,None,:,:,None].flip(-1)
#     grid1 = get_space_grid(nH1,nW1,dtype=th.float,device="cuda")
#     grid1 = grid1[:,None,:,:,None].flip(-1)

#     # -- add grid0 --
#     inds = flow.clone()
#     inds[...,1:] = flow[...,1:] + grid0
#     inds[...,1:] = flow[...,1:] + grid0

#     flows = run(flows,flows,labels,stride0,stride1)
#     inds = flow2inds(flows,stride1)
#     flows_k = inds2flows(inds,stride0)

#     return flows_k

def run_topk(weights,flows_k,K,descending=True):

    # -- reshape --
    B,HD,Q,S,_ = flows_k.shape
    weights = rearrange(weights,'b hd q s -> (b hd q) s')
    flows_k = rearrange(flows_k,'b hd q s tr -> (b hd q) s tr')
    # names = rearrange(names,'b hd s t nh nw tw -> (b hd t nh nw) s tw')
    device = weights.device

    # -- get ordering --
    order = th.argsort(weights,-1,descending=descending)[:,:K]

    # -- get topk --
    weights = th.gather(weights,-1,order)

    flows_topk = -th.inf*th.ones(weights.shape+(3,),device=device,dtype=flows_k.dtype)
    for i in range(flows_k.shape[-1]):
        flows_topk[...,i] = th.gather(flows_k[...,i],-1,order)

    # names_topk = th.zeros(weights.shape+(2,),device=device,dtype=weights.dtype)
    # for i in range(names.shape[-1]):
    #     names_topk[...,i] = th.gather(names[...,i],-1,order)

    # -- unpack --
    weights = rearrange(weights,'(b hd q) k -> b hd q k',b=B,hd=HD)
    flows_topk = rearrange(flows_topk,'(b hd q) k tr -> b hd q k tr',b=B,hd=HD)
    flows_topk = flows_topk.type(flows_k.dtype)

    return weights,flows_topk
