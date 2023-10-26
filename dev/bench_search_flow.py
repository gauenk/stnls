"""

Check runtimes & memory

"""

# -- basic --
import os
import torch as th
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict

# -- plot --
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# -- caching --
import cache_io

# -- stnls --
import stnls

# -- spynet --
from nlnet.original.spynet import SpyNet

# -- bench --
from dev_basics.utils.misc import set_seed
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt
from dev_basics import flow


def run_spynet_grid(vid,wt,stride0):
    net = SpyNet("./weights/spynet/spynet_sintel_final-3d2a1287.pth")
    W_t = 2*wt
    flows = []
    for ti in range(T):
        for tj in range(W_t):
            flow_ij = net(vid[:,ti],vid[:,tj])
            flows.append(flow_ij)
    # ...
    # benchmark spynet for 1st order, 2nd order, and 3rd order flows.

def run_acc_pytorch(fflow,bflow,wt,stride0):
    return run_acc(fflow,bflow,wt,stride0,"pytorch")

def run_acc_stnls(fflow,bflow,wt,stride0):
    return run_acc(fflow,bflow,wt,stride0,"stnls")

def run_acc(fflow,bflow,wt,stride0,fwd_mode):
    aflows = stnls.nn.accumulate_flow(fflow,bflow,fwd_mode=fwd_mode)
    extract = stnls.nn.extract_search_from_accumulated
    flows = extract(aflows.fflow,aflows.bflow,wt,stride0)
    return flows

def run_stnls(fflow,bflow,wt,stride0):
    return stnls.nn.search_flow(fflow,bflow,wt,stride0)

def get_fxn(name):
    if name == "acc_pytorch":
        return run_acc_pytorch
    elif name == "acc_stnls":
        return run_acc_stnls
    elif name == "stnls":
        return run_stnls
    else:
        raise ValueError(f"Uknown function name [{name}]")

def exec_method(cfg):

    # -- init --
    timer = ExpTimer()
    memer = GpuMemer()
    set_seed(cfg.seed)

    # -- allocate --
    B,T,H,W = cfg.B,cfg.T,cfg.H,cfg.W
    fflow = th.zeros((B,T,2,H,W),device="cuda").requires_grad_(True)
    bflow = th.zeros((B,T,2,H,W),device="cuda").requires_grad_(True)

    # -- get function --
    fxn = get_fxn(cfg.name)

    # -- compute forward --
    with TimeIt(timer,"fwd"):
        with MemIt(memer,"fwd"):
            flows = fxn(fflow,bflow,cfg.wt,cfg.stride0)
    print(flows.shape)

    # -- compute backward --
    grad = th.randn_like(flows)
    with TimeIt(timer,"bwd"):
        with MemIt(memer,"bwd"):
            th.autograd.backward(flows,grad)

    results = {"fwd":timer['fwd'],
               "bwd":timer['bwd'],
               "fwd_res":memer['fwd']['res'],
               "fwd_alloc":memer['fwd']['alloc'],
               "bwd_res":memer['bwd']['res'],
               "bwd_alloc":memer['bwd']['alloc'],
    }
    return results



def main():

    # -- print pid --
    print("PID: ",os.getpid())
    cfg = edict({"B":1,"T":10,"H":256,"W":256,"wt":1,"stride0":1,"seed":0})
    exps_cfg = {"cfg":cfg,
                "group0":{"name":["acc_pytorch","acc_stnls","stnls"]}}
    exps = cache_io.exps.unpack(exps_cfg)

    # -- run exps -
    df = cache_io.run_exps(exps,exec_method,clear=False,
                           name = ".cache_io/bench_search_flow/",
                           skip_loop=False,clear_fxn=None,
                           records_reload=False,to_records_fast=False,
                           use_wandb=False,enable_dispatch="slurm",
                           records_fn=".cache_io_pkl/bench_search_flow")
    print(df[['name','fwd','bwd','fwd_alloc','bwd_alloc']])


if __name__ == "__main__":
    main()
