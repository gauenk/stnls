
# -- python --
import torch as th
from functools import partial
from easydict import EasyDict as edict

# -- cpp cuda kernel --
import dnls_cuda

def init():
    return run

def run(*args,**kwargs):
    if len(args) == 1:
        return run_flows(*args,**kwargs)
    elif len(args) == 2:
        return run_pair(*args,**kwargs)
    elif len(args) == 3:
        return run_pair(*args,**kwargs)

def run_flows(flows,stride0=1):
    return run_pair(flows.fflow,flows.bflow,stride0=stride0)

def run_pair(fflow,bflow,stride0=1):

    # -- unpack --
    B,T,_,H,W = fflow.shape
    B,T,_,H,W = bflow.shape

    # -- get size --
    nH = (H-1)//stride0+1
    nW = (W-1)//stride0+1

    # -- allocate --
    pfflow = th.zeros((B,T-1,T,2,nH,nW),device=fflow.device,
                      dtype=fflow.dtype)
    pbflow = th.zeros((B,T-1,T,2,nH,nW),device=bflow.device,
                      dtype=bflow.dtype)
    # -- run --
    dnls_cuda.optical_flow_accumulate(fflow,bflow,pfflow,pbflow,stride0)

    # -- format --
    flows = edict()
    flows.fflow = pfflow
    flows.bflow = pbflow

    return flows

