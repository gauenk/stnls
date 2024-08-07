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

class scatter_tensor(th.autograd.Function):

    @staticmethod
    def forward(ctx,tensor,flows_k,labels,stride0,stride1,H,W,invalid=th.inf):
        ctx_vars = {"tensor_shape":tensor.shape,"flows_k":flows_k,"labels":labels,
                    "stride0":stride0,"stride1":stride1,"H":H,"W":W,"invalid":invalid}
        for name,val in ctx_vars.items():
            setattr(ctx,name,val)
        return run(tensor,flows_k,labels,stride0,stride1,H,W,invalid)

    @staticmethod
    def backward(ctx, grad_scatter_tensor):
        # -- unpack --
        labels = ctx.labels
        flows_k = ctx.flows_k
        stride0 = ctx.stride0
        device = grad_scatter_tensor.device
        dtype = grad_scatter_tensor.dtype
        H,W = ctx.H,ctx.W

        # -- get shapes --
        # print(ctx.tensor_shape,grad_scatter_tensor.shape,labels.shape)
        B,HD,T,nH,nW,S = ctx.tensor_shape
        Q = T*nH*nW # shape = (B,HD,Q,S,M)

        # -- reshape --
        tensor_grad = th.zeros((B,HD,Q,S,1),device=device,dtype=dtype)
        grad_scatter_tensor = grad_scatter_tensor[...,None]
        # grad_scatter_tensor = th.randn_like(grad_scatter_tensor)

        # -- bwd --
        # print(tensor_grad.shape,grad_scatter_tensor.shape,labels.shape,flows_k.shape)
        stnls_cuda.scatter_tensor_backward(tensor_grad,grad_scatter_tensor,
                                           labels,flows_k,stride0,H,W)
        tensor_grad = tensor_grad[...,0].reshape(ctx.tensor_shape)
        # print(tensor_grad)
        # print(tensor_grad.norm())

        return tensor_grad,None,None,None,None,None,None,None

def apply(tensor,flows_k,labels,stride0,stride1,H,W,invalid=th.inf):
    return scatter_tensor.apply(tensor,flows_k,labels,stride0,stride1,H,W,invalid)

def run(tensor,flows_k,labels,stride0,stride1,H,W,invalid=th.inf):

    # -- unpack shapes --
    B,HD,T,nH0,nW0,K = tensor.shape[:6]
    Q0 = T*nH0*nW0
    S = labels.max().long().item()+1
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
    scatter_tensor = invalid*th.ones(shape,device=labels.device,dtype=tensor.dtype)
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

def run_topk(weights,flows_k,labels,K,descending=True):

    # -- reshape --
    B,HD,Q,S,_ = flows_k.shape
    weights = rearrange(weights,'b hd q s -> (b hd q) s')
    flows_k = rearrange(flows_k,'b hd q s tr -> (b hd q) s tr')
    labels = rearrange(labels,'b hd q s -> (b hd q) s')
    # names = rearrange(names,'b hd s t nh nw tw -> (b hd t nh nw) s tw')
    device = weights.device
    if K <= 0: K = S

    # -- get ordering --
    order = th.argsort(weights,-1,descending=descending)[:,:K]

    # -- get topk --
    weights = th.gather(weights,-1,order)
    labels = th.gather(labels,-1,order)

    flows_topk = -th.inf*th.ones(weights.shape+(3,),device=device,dtype=flows_k.dtype)
    for i in range(flows_k.shape[-1]):
        flows_topk[...,i] = th.gather(flows_k[...,i],-1,order)

    # names_topk = th.zeros(weights.shape+(2,),device=device,dtype=weights.dtype)
    # for i in range(names.shape[-1]):
    #     names_topk[...,i] = th.gather(names[...,i],-1,order)

    # -- unpack --
    weights = rearrange(weights,'(b hd q) k -> b hd q k',b=B,hd=HD)
    labels = rearrange(labels,'(b hd q) k -> b hd q k',b=B,hd=HD)
    flows_topk = rearrange(flows_topk,'(b hd q) k tr -> b hd q k tr',b=B,hd=HD)
    flows_topk = flows_topk.type(flows_k.dtype)

    return weights,flows_topk,labels
