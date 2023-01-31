
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
SAVE_DIR = Path("./output/tests/prod_search_with_heads")

def pytest_generate_tests(metafunc):
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)
    test_lists = {"ps":[7],"stride0":[4],"stride1":[1],
                  "dilation":[1],"wt":[0],"ws":[23],"k":[15],
                  "ws_r":[3],"k_r":[7],
                  "exact":[True],"nheads":[4],"seed":[0],"anchor_self":[True]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def test_fwd(k_r,ws_r,ws,wt,k,ps,stride0,stride1,dilation,nheads,exact,seed):
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
    anchor_self = True
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    reflect_bounds = True
    use_k = k > 0
    use_adj = False
    adj = 0
    use_self = anchor_self

    # -- load data --
    vid = dnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid.to(device)[:,:5,].contiguous()
    vid = repeat(vid,'b t c h w -> b t (r c) h w',r=12)[:,:32].contiguous()
    vid /= vid.max()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- compute flow --
    flows = dnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)
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
    search_gt = dnls.search.NonLocalSearch(ws, wt, ps, k, nheads,
                                 dilation=dil,stride0=stride0, stride1=stride1,
                                 reflect_bounds=reflect_bounds,full_ws=False,
                                 anchor_self=anchor_self,remove_self=False)
    # search_gt = dnls.search_dev.init("prod_with_heads",flows.fflow, flows.bflow,
    #                                  k, ps, pt, ws, wt, nheads,
    #                                  chnls=-1,dilation=dil,
    #                                  stride0=stride0, stride1=stride1,
    #                                  reflect_bounds=reflect_bounds,use_k=use_k,
    #                                  h0_off=h0_off, w0_off=w0_off,
    #                                  h1_off=h1_off, w1_off=w1_off,
    #                                  search_abs=False,use_adj=use_adj,
    #                                  anchor_self=anchor_self,use_self=use_self,
    #                                  exact=exact)


    # -- [gt] search --
    # dists_gt,inds_gt = search_gt(vid,vid)
    dists_gt,inds_gt = search_gt(vid,vid,flows.fflow,flows.bflow)

    #
    # -- [te] search --
    #

    # -- search big stride --
    scale = 2
    stride0_s = scale * stride0
    search_gt.stride0 = stride0_s
    # nH,nW = (H-1)//stride0_s+1,(W-1)//stride0_s+1
    # ntotal = T * nH * nW
    # _,inds_te = search_gt(vid,vid)
    _,inds_te = search_gt(vid,vid,flows.fflow,flows.bflow)

    # -- upsample --
    inds_te = dnls.nn.interpolate_inds(inds_te,scale,stride0,T,H,W)

    # -- reshape --
    inds_gt = rearrange(inds_gt,'b hd (t h w) k tr -> b hd t h w k tr',t=T,h=nH)
    inds_te = rearrange(inds_te,'b hd (t h w) k tr -> b hd t h w k tr',t=T,h=nH)

    # -- eq and neq ind args --
    print("inds_gt.shape: ",inds_gt.shape)
    print("inds_te.shape: ",inds_te.shape)
    print("-"*20)
    print(inds_gt[0,0,0,:3,:3,0])
    print(inds_te[0,0,0,:3,:3,0])
    # print(inds_gt[0,0,0,:3,:10])
    # print(inds_te[0,0,0,:3,:10])
    # print(inds_gt[0,0,0,:20,0])
    # print(inds_te[0,0,0,:20,0])
    print("-"*20)

    #
    # -- equal inds @ k == 0 --
    #

    # diff = th.sum(th.abs(inds_gt[...,0,:] - inds_te[...,::2,0,:]))
    diff = th.sum(th.abs(inds_gt[...,0,:] - inds_te[...,:,0,:]))
    assert diff < 1e-10


