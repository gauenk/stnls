
# -- python --
import sys

# -- data mgnmt --
from pathlib import Path
from easydict import EasyDict as edict

# -- testing --
import pytest
import random

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- dnls --
import dnls
import dnls.utils.gpu_mem as gpu_mem
from dnls.utils.pads import comp_pads
from dnls.utils.inds import get_batching_info

# -- meshgrid --


# -- test func --
from torch.nn.functional import fold,unfold,pad
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/non_local_search")

def pytest_generate_tests(metafunc):
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)
    test_lists = {"ps":[7],"stride0":[4],"stride1":[1],
                  "dilation":[1],"wt":[1],"ws":[1], "wr":[13],"kr":[-1],
                  "k":[3],"scale":[2],"exact":[True],"nheads":[1],
                  "seed":[0]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def test_fwd(kr,wr,scale,ws,wt,k,ps,stride0,stride1,dilation,nheads,exact,seed):
    """

    Test the CUDA code with torch code

    Forward Pass

    """


    # -- get args --
    dil = dilation
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    pt = 1
    set_seed(seed)

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    reflect_bounds = False
    use_k = k > 0
    use_adj = False
    adj = 0
    search_abs = ws == -1
    anchor_self = True
    use_self = anchor_self
    rbwd = True
    nbwd = 1

    # -- load data --
    vid = dnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid.to(device)[:,:5,].contiguous()
    vid = repeat(vid,'b t c h w -> b t (r c) h w',r=12)[:,:32].contiguous()
    vid /= vid.max()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- compute flow --
    flows = dnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)
    flows.fflow = (2*th.randn_like(flows.fflow)).clamp(-3,3)
    flows.bflow = (2*th.randn_like(flows.bflow)).clamp(-3,3)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape

    # -- exec fold fxns --
    search_gt = dnls.search.NonLocalSearch(ws, wt, ps, k, nheads,
                                 dilation=dil,stride0=stride0, stride1=stride1,
                                 reflect_bounds=reflect_bounds,full_ws=False,
                                 anchor_self=anchor_self,remove_self=False,
                                 use_adj=use_adj,rbwd=rbwd,nbwd=nbwd,exact=exact)
    search_te = dnls.search.ApproxSpaceSearch(ws, wt, ps, k, wr, kr, scale, nheads,
                                 dilation=dil,stride0=stride0, stride1=stride1,
                                 reflect_bounds=reflect_bounds,full_ws=False,
                                 anchor_self=anchor_self,remove_self=False,
                                 use_adj=use_adj,rbwd=rbwd,nbwd=nbwd,exact=exact)

    # -- test api --
    dists_gt,inds_gt = search_gt(vid,vid,flows.fflow,flows.bflow)
    dists_te,inds_te = search_te(vid,vid,flows.fflow,flows.bflow)
    th.cuda.synchronize()

    # -- compare --
    assert th.all(dists_te >= dists_gt)



