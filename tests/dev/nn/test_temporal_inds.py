
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

# -- stnls --
import stnls
import stnls.utils.gpu_mem as gpu_mem
from stnls.utils.pads import comp_pads
from stnls.utils.inds import get_batching_info

# -- meshgrid --

# -- test func --
from torch.nn.functional import fold,unfold,pad
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/prod_search_with_heads")

def pytest_generate_tests(metafunc):
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)
    test_lists = {"ps":[7],"stride0":[4],"stride1":[1],
                  "dilation":[1],"wt":[0],"ws":[23],"k":[15],
                  "ws_r":[3],"k_r":[7],
                  "exact":[True],"nheads":[4],"seed":[0],
                  "anchor_self":[False]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def test_fwd(k_r,ws_r,ws,wt,k,ps,stride0,stride1,dilation,nheads,anchor_self,exact,seed):
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
    reflect_bounds = True
    use_k = k > 0
    use_adj = False
    adj = 0
    anchor_self = anchor_self
    use_self = anchor_self

    # -- load data --
    vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid.to(device)[:,:5,].contiguous()
    vid = repeat(vid,'b t c h w -> b t (r c) h w',r=12)[:,:32].contiguous()
    vid /= vid.max()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)
    flows.fflow = 10*th.randn_like(flows.fflow)
    flows.bflow = 10*th.randn_like(flows.bflow)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    B,T,color,H,W = shape
    vshape = vid.shape
    chnls = vid.shape[-3]
    # B,T,C,H,W = vid.shape

    # -- batching info --
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    ntotal = T * nH * nW
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1
    h0_off,w0_off = 0,0
    h1_off,w1_off = 0,0

    # -- exec fold fxns --
    search = stnls.search.NonLocalSearch(ws, 0, ps, k, nheads,
                                        dilation=dil,stride0=stride0, stride1=stride1,
                                        reflect_bounds=reflect_bounds,full_ws=False,
                                        anchor_self=anchor_self,remove_self=False,
                                        use_adj=use_adj,rbwd=rbwd,nbwd=nbwd,exact=exact)
    dists,inds = search(vid,vid,flows.fflow,flows.bflow)

    # -- apply temporal inds --
    inds_t = stnls.nn.temporal_inds(inds,wt,flows.fflow,flows.bflow)
