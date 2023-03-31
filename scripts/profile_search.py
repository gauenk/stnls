# -- misc --
import cv2,tqdm,copy
import numpy as np

import tempfile
import sys
import shutil
from pathlib import Path
from easydict import EasyDict as edict

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- dnls --
import dnls
import dnls.utils.gpu_mem as gpu_mem
from dnls.utils.pads import comp_pads
from dnls.utils.inds import get_batching_info

def main():

    # -- fix consts --
    ps,k = 7,10
    stride0,stride1 = 1,1
    dilation = 1
    reflect_bounds,exact = True,False

    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    k,pt = 1,1
    wt = 3
    ws = -1
    # stride0 = stride
    # stride1 = 1
    search_abs = k<=0
    use_k = k>0
    if ws == -1 and k > 0: ws = 10
    exact = True
    use_adj = True

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    adj = True
    only_full = True

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device)[None,:2].contiguous()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- compute flow --
    flows = dnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)

    # -- search --
    search = dnls.search.NonLocalSearch(ws,wt,ps,k,nheads=1,
                                        stride0=stride0,dist_type="l2")

    # -- profile --
    with th.autograd.profiler.emit_nvtx():
        score_te,inds_te = search(vid,vid,flows.fflow,flows.bflow)

if __name__ == "__main__":
    main()
