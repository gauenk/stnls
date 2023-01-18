
# -- python --
import torch as th
from functools import partial

# -- cpp cuda kernel --
import dnls_cuda

def init():
    return run

def run(fflow,bflow,stride0=1):
    B,T,_,H,W = fflow.shape
    B,T,_,H,W = bflow.shape
    pfflow = th.zeros((B,T-1,T,2,H,W),device=fflow.device,
                      dtype=fflow.dtype)
    pbflow = th.zeros((B,T-1,T,2,H,W),device=bflow.device,
                      dtype=bflow.dtype)
    dnls_cuda.optical_flow_accumulate(fflow,bflow,pfflow,pbflow,stride0)
    return pfflow,pbflow

