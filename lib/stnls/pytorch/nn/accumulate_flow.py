
# -- python --
import torch as th
from functools import partial
from easydict import EasyDict as edict

# -- cpp cuda kernel --
import stnls_cuda

def init():
    return run

def run(*args,**kwargs):
    if len(args) == 1:
        return run_flows(*args,**kwargs)
    elif len(args) == 2:
        return run_pair(*args,**kwargs)
    elif len(args) == 3:
        return run_pair(*args,**kwargs)

def run_flows(flows,stride0=1,dtype=None):
    return run_pair(flows.fflow,flows.bflow,stride0=stride0,dtype=dtype)

def run_pair(fflow,bflow,stride0=1,dtype=None):

    # -- unpack --
    B,T,_,H,W = fflow.shape
    B,T,_,H,W = bflow.shape
    device = fflow.device
    dtype = fflow.dtype if dtype is None else dtype

    # -- get size --
    nH = (H-1)//stride0+1
    nW = (W-1)//stride0+1

    # -- allocate --
    pfflow = th.zeros((B,T,T-1,2,nH,nW),device=device,dtype=dtype)
    pbflow = th.zeros((B,T,T-1,2,nH,nW),device=device,dtype=dtype)
    # -- run --
    stnls_cuda.accumulate_flow(fflow,bflow,pfflow,pbflow,stride0)

    # -- format --
    flows = edict()
    flows.fflow = pfflow
    flows.bflow = pbflow

    return flows

