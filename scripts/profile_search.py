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
    vid = th.from_numpy(vid).to(device)[:2].contiguous()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],-2)

    # -- normalize --
    vid /= vid.max()
    # vidr = th.ones_like(vid)
    vidr = th.rand_like(vid)

    # -- compute flow --
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape
    chnls = vid.shape[1]

    # -- batching --
    _,_,n0,n1 = get_batching_info(vid.shape,stride0,stride1,ps,dil)
    n_h0,n_w0 = n0[0],n0[1]
    n_h1,n_w1 = n1[0],n1[1]

    # -- exec fold fxns --
    h0_off,w0_off,_,_ = comp_pads(vid.shape, ps, stride0, 1)
    h1_off,w1_off,_,_ = comp_pads(vid.shape, ps, stride1, 1)
    search = dnls.search.init("l2_with_index",
                              flows.fflow, flows.bflow, k, ps, pt,
                              ws, wt, dilation=dil,
                              stride0=stride0, stride1=stride1,
                              use_k = use_k,use_adj=use_adj,
                              reflect_bounds=reflect_bounds,
                              search_abs=search_abs,exact=exact,
                              h0_off=h0_off,w0_off=w0_off,
                              h1_off=h1_off,w1_off=w1_off)

    # -- batching info --
    n_h0,n_w0 = search.query_batch_info(vid.shape) # just showing api
    ntotal = t * n_h0 * n_w0
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- query inds --
    qindex = 0

    # -- run search --
    with th.autograd.profiler.emit_nvtx():
        score_te,inds_te = search(vid,qindex,ntotal,vid1=vidr)

if __name__ == "__main__":
    main()
