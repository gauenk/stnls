"""

The search method based on N3Net's MatMult

"""

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

# -- meshgrid --


# -- test func --
from torch.nn.functional import fold,unfold,pad
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/n3mm_search")

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_data(dnames,ext,device="cuda:0"):
    vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid.to(device)[:,:5,].contiguous()
    vid = repeat(vid,'b t c h w -> b t (r c) h w',r=12)[:,:32].contiguous()
    vid = vid[:1,:,:1].contiguous()
    vid /= vid.max()
    return vid

def pytest_generate_tests(metafunc):
    # only for stride1 == 1
    test_lists = {"wt":[1],"ws":[5],"k":[10],"ps":[3],
                  "stride0":[1],"stride1":[1],"dilation":[1],
                  "nheads":[1],
                  "dist_type":["prod"],"seed":[0]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def test_fwd(ws,wt,k,ps,stride0,stride1,dilation,
             nheads,dist_type,seed):
    """

    Test the CUDA code with torch code

    Forward Pass

    """


    # -- get args --
    pt = 1
    dil = dilation
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    device = "cuda:0"
    clean_flow = True
    run_flow = False
    reflect_bounds = True
    self_action = None
    set_seed(seed)

    # -- load data --
    vid = get_data(dnames,ext)
    vid0 = th.rand_like(vid)-0.5
    vid1 = th.rand_like(vid)-0.5

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(run_flow,clean_flow,vid,vid,0.)
    flows.fflow = 2*(th.rand_like(flows.fflow)-0.5).round()
    flows.bflow = 2*(th.rand_like(flows.bflow)-0.5).round()
    # float flows won't match since search inds for n3mm summed with
    # the search grid and THEN "int-ed"
    # while the non-local search computes the "int" of the _flow_ before searching.

    # -- exec fold fxns --
    sch = stnls.search
    search_te = sch.N3MatMultSearch(ws, wt, ps, k, nheads,
                                    dist_type=dist_type, dilation=dil,
                                    stride0=stride0, stride1=stride1,
                                    reflect_bounds=reflect_bounds,
                                    self_action=self_action)
    search_gt = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                   dist_type=dist_type, dilation=dil,
                                   stride0=stride0, stride1=stride1,
                                   reflect_bounds=reflect_bounds,
                                   self_action=self_action,itype="int")

    # -- [testing] search --
    dists_te,inds_te = search_te(vid0,vid1,flows.fflow,flows.bflow)
    inds_te = stnls.utils.inds2flow(inds_te,stride0)

    # -- [groundtruth] search --
    dists_gt,inds_gt = search_gt(vid0,vid1,flows.fflow,flows.bflow)

    # -- compare --
    assert th.allclose(dists_te,dists_gt,1e-3,1e-3,equal_nan=True)
    assert th.allclose(inds_te,inds_gt,1e-3,1e-3,equal_nan=True)

def test_bwd(ws,wt,k,ps,stride0,stride1,dilation,
             nheads,dist_type,seed):
    """

    Test the CUDA code with torch code

    Forward Pass

    """

    # -- get args --
    dil = dilation
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    pt = 1
    device = "cuda:0"
    clean_flow = True
    run_flow = False
    reflect_bounds = True
    self_action = None
    set_seed(seed)

    # -- load data --
    B,T,HD,F,H,W = 1,3,nheads,1,32,32
    vid = th.ones((B,T,HD*F,H,W),device=device)
    vid0 = th.rand_like(vid)-0.5
    vid1 = th.rand_like(vid)-0.5

    # -- compute flow --
    flows = edict()
    flows.fflow = th.ones((B,T,2,H,W)).cuda()/2.
    flows.bflow = th.ones((B,T,2,H,W)).cuda()/2.
    flows.fflow = th.clamp(10*th.zeros_like(flows.fflow),-10,10)
    flows.bflow = th.clamp(10*th.zeros_like(flows.bflow),-10,10)
    # flows.fflow = 2*(th.rand_like(flows.fflow)-0.5).round()
    # flows.bflow = 2*(th.rand_like(flows.bflow)-0.5).round()

    # -- allow grads --
    vid_te0,vid_te1 = vid.clone(),vid.clone()
    vid_te0[...] = 2
    vid_te1[...] = 1
    vid_te0.requires_grad_(True)
    vid_te1.requires_grad_(True)
    vid_gt0,vid_gt1 = vid.clone(),vid.clone()
    vid_gt0[...] = 2
    vid_gt1[...] = 1
    vid_gt0.requires_grad_(True)
    vid_gt1.requires_grad_(True)

    # -- exec fold fxns --
    sch = stnls.search
    search_te = sch.N3MatMultSearch(ws, wt, ps, -1, nheads,
                                    dist_type=dist_type, dilation=dil,
                                    stride0=stride0, stride1=stride1,
                                    reflect_bounds=reflect_bounds,
                                    self_action=self_action)
    search_gt = sch.NonLocalSearch(ws, wt, ps, -1, nheads,
                                   dist_type=dist_type, dilation=dil,
                                   stride0=stride0, stride1=stride1,
                                   reflect_bounds=reflect_bounds,
                                   self_action=self_action,itype="int")

    # -- [testing] search --
    dists_te,inds_te = search_te(vid_te0,vid_te1,flows.fflow,flows.bflow)
    inds_te = stnls.utils.inds2flow(inds_te,stride0)
    th.cuda.synchronize()

    # -- [groundtruth] search --
    dists_gt,inds_gt = search_gt(vid_gt0,vid_gt1,flows.fflow,flows.bflow)
    th.cuda.synchronize()

    # -- compare --
    assert th.allclose(dists_te,dists_gt,1e-3,1e-3,equal_nan=True)
    assert th.allclose(inds_te,inds_gt,1e-3,1e-3,equal_nan=True)

    # -- compute bwd --
    dists_grad = 2*(th.rand_like(dists_te)-0.5)
    th.autograd.backward(dists_te,dists_grad)
    th.autograd.backward(dists_gt,dists_grad)

    # -- for both grads --
    sH,sW = ps,ps
    eH,eW = H-ps,W-ps
    _grads_te = [vid_te0.grad,vid_te1.grad]
    _grads_gt = [vid_gt0.grad,vid_gt1.grad]
    for idx,(grads_te,grads_gt) in enumerate(zip(_grads_te,_grads_gt)):
        te = grads_te[...,sH:eH,sW:eW]
        gt = grads_gt[...,sH:eH,sW:eW]
        # print(te[0,0,0,:3,:3])
        # print(gt[0,0,0,:3,:3])
        assert th.allclose(te,gt,1e-3,1e-3)
