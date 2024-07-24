
"""

wt = temporal window
stride0 = query stride for flows; fflow.shape[-2:] = (nH,nW) nH = (H-1)//stride0+1

flows = stnls.nn.search_flow(fflow,bflow,wt,stride0)

"""


# -- python --
import torch as th
from functools import partial
from easydict import EasyDict as edict
import torch.nn.functional as F

# -- cpp cuda kernel --
import stnls_cuda

def init():
    return run

def run(fflow,bflow,wt,stride0):

    # -- exec --
    if wt > 0:
        fflow = fflow.contiguous()
        bflow = bflow.contiguous()
        flows = search_flow_th.apply(fflow,bflow,wt,stride0)
    else:
        flows = empty_flows(fflow,wt,stride0)

    return flows

def empty_flows(fflow,wt,stride0):
    B,T,_,H,W = fflow.shape
    nH = (H-1)//stride0+1
    nW = (W-1)//stride0+1
    W_t = min(2*wt+1,T)
    flows = th.zeros((B,T,W_t-1,2,nH,nW),
                     device=fflow.device,dtype=fflow.dtype)
    return flows

class search_flow_th(th.autograd.Function):

    @staticmethod
    def forward(ctx, fflow, bflow, wt, stride0):

        # -- allocate --
        B,T,_,H,W = fflow.shape
        nH = (H-1)//stride0+1
        nW = (W-1)//stride0+1
        W_t = min(2*wt+1,T)
        flows = th.zeros((B,T,W_t-1,2,nH,nW),
                         device=fflow.device,dtype=fflow.dtype)

        # -- forward --
        stnls_cuda.search_flow_forward(fflow,bflow,flows,wt,stride0)

        # -- setup ctx --
        ctx.save_for_backward(fflow,bflow,flows)
        ctx_vars = {"wt":wt,"stride0":stride0,"fshape":list(fflow.shape)}
        for name,val in ctx_vars.items():
            setattr(ctx,name,val)

        return flows

    @staticmethod
    def backward(ctx, grad_flows):

        # -- init --
        grad_fflow = th.zeros(ctx.fshape,device=grad_flows.device)
        grad_bflow = th.zeros(ctx.fshape,device=grad_flows.device)

        # -- get sizes --
        wt = ctx.wt
        stride0 = ctx.stride0
        dtype = grad_fflow.dtype
        device = grad_fflow.device
        B,T,_,H,W = grad_fflow.shape
        nH = (H-1)//stride0+1
        nW = (W-1)//stride0+1
        # dev = th.zeros((B,T*nH*nW,T-1,T-1,2,2,6),device=device,dtype=dtype)

        # -- backward --
        fflow,bflow,flows = ctx.saved_tensors
        # bflow = bflow.flip(1)
        stnls_cuda.search_flow_backward(grad_fflow,grad_bflow,
                                        grad_flows,fflow,bflow,flows,
                                        ctx.wt,ctx.stride0)
        # grad_bflow = grad_bflow.flip(1)
        # print("none check: ",grad_fflow is None,grad_bflow is None)

        return grad_fflow,grad_bflow,None,None

