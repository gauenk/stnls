
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
SAVE_DIR = Path("./output/tests/non_local_search")

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_data(dnames,ext="jpg",device="cuda:0"):
    vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid.to(device)[:,:3,].contiguous()
    vid = repeat(vid,'b t c h w -> b t (r c) h w',r=12)[:,:,:3].contiguous()
    vid /= vid.max()
    return vid

def pytest_generate_tests(metafunc):
    test_lists = {"ps":[1],"stride0":[4],"stride1":[1],
                  "dilation":[1],"wt":[1],"ws":[3],
                  "k":[-1],"exact":[False],"nheads":[1],
                  "self_action":[None],"seed":[1],"dist_type":["prod"],
                  "k_agg":[-1]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)


def test_fwd_n3mm(ws,wt,k,ps,stride0,stride1,dilation,k_agg,
                  nheads,self_action,exact,dist_type,seed):

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
    use_adj = False # keep false since unfold/fold doesn't match search
    set_seed(seed)

    # -- load data --
    vid = get_data(dnames,ext)

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(run_flow,clean_flow,vid,vid,0.)
    flows.fflow = th.clamp(10.*th.randn_like(flows.fflow),-3,3).round()
    flows.bflow = th.clamp(10.*th.randn_like(flows.bflow),-3,3).round()

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape

    # -- exec fold fxns --
    sch = stnls.search
    search_te = sch.N3MatMultSearch(ws, wt, ps, k, nheads,
                                    dist_type=dist_type, dilation=dil,
                                    stride0=stride0, stride1=stride1,
                                    reflect_bounds=reflect_bounds,
                                    self_action=self_action,
                                    use_adj=use_adj)
    search_gt = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                   dist_type=dist_type, dilation=dil,
                                   stride0=stride0, stride1=stride1,
                                   reflect_bounds=reflect_bounds,
                                   self_action=self_action,
                                   use_adj=use_adj,normalize_bwd=True,
                                   itype="int",full_ws=True)

    # -- [testing] search --
    dists_te,inds_te = search_te(vid,vid,flows.fflow,flows.bflow)
    th.cuda.synchronize()

    # -- [groundtruth] search --
    dists_gt,inds_gt = search_gt(vid,vid,flows.fflow,flows.bflow)
    th.cuda.synchronize()

    # -- pick tolerance --
    if dist_type == "prod":
        mean_tol = 1e-5
        max_tol = 1e-5
    else:
        mean_tol = 1e-3
        max_tol = 1e-1

    # -- compare --
    isinf = th.isinf(dists_gt)
    issmall = dists_gt < 1e-4
    args0 = th.where(th.logical_not(th.logical_or(isinf,issmall))) # remove invalid
    diff = th.abs(dists_te - dists_gt) / (dists_gt.abs()+1e-8)

    # -- test --
    error = diff[args0].mean().item()
    if error > mean_tol: print("error: ",error)
    assert error < mean_tol

    max_error = diff[args0].max().item()
    if max_error > max_tol: print("max error: ",max_error)
    assert max_error < max_tol

def test_bwd_n3mm(ws,wt,k,ps,stride0,stride1,dilation,
                  k_agg,nheads,self_action,dist_type,seed):
    """

    Test the CUDA code with torch code

    Forward Pass

    """

    # -- get args --
    dil = dilation
    pt = 1
    device = "cuda:0"
    clean_flow = True
    run_flow = False
    reflect_bounds = True
    use_adj = False # keep false since unfold/fold doesn't match search
    self_action = None

    # -- load data --
    set_seed(seed)
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    vid = get_data(dnames)

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(run_flow,clean_flow,vid,vid,0.)
    flows.fflow = th.clamp(10.*th.randn_like(flows.fflow),-10,10).round()
    flows.bflow = th.clamp(10.*th.randn_like(flows.bflow),-10,10).round()

    # -- allow grads --
    vid_te0,vid_te1 = vid.clone(),vid.clone()
    vid_te0.requires_grad_(True)
    vid_te1.requires_grad_(True)
    vid_gt0,vid_gt1 = vid.clone(),vid.clone()
    vid_gt0.requires_grad_(True)
    vid_gt1.requires_grad_(True)

    # -- exec fold fxns --
    sch = stnls.search
    search_gt = sch.N3MatMultSearch(ws, wt, ps, k, nheads,
                                    dist_type=dist_type, dilation=dil,
                                    stride0=stride0, stride1=stride1,
                                    reflect_bounds=reflect_bounds,
                                    self_action=self_action,use_adj=use_adj,
                                    normalize_bwd=True)
    search_te = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                   dist_type=dist_type, dilation=dil,
                                   stride0=stride0, stride1=stride1,
                                   reflect_bounds=reflect_bounds,
                                   self_action=self_action,use_adj=use_adj,
                                   normalize_bwd=True,itype="int")
    k_agg = k# if k_agg <= 0 else k_agg

    # -- [testing] search --
    dists_te,inds_te = search_te(vid_te0,vid_te1,flows.fflow,flows.bflow)
    th.cuda.synchronize()

    # -- [groundtruth] search --
    dists_gt,inds_gt = search_gt(vid_gt0,vid_gt1,flows.fflow,flows.bflow)
    th.cuda.synchronize()

    # -- pick tolerance --
    if dist_type == "prod":
        mean_tol = 1e-5
        max_tol = 1e-5
    else:
        mean_tol = 1e-3
        max_tol = 1e-1

    # -- compare --
    isinf = th.isinf(dists_gt)
    issmall = dists_gt < 1e-4
    args0 = th.where(th.logical_not(th.logical_or(isinf,issmall))) # remove invalid
    diff = th.abs(dists_te - dists_gt) / (dists_gt.abs()+1e-5)
    args1 = th.where(diff>1-3)

    error = diff[args0].mean().item()
    if error > mean_tol: print("error: ",error)
    assert error < mean_tol

    max_error = diff[args0].max().item()
    if max_error > max_tol: print("max error: ",max_error)
    assert max_error < max_tol

    # -- compute bwd --
    # dists_grad = th.randn_like(dists_te)
    dists_grad = th.ones_like(dists_te)
    th.autograd.backward(dists_te,dists_grad)
    th.autograd.backward(dists_gt,dists_grad)

    # -- for both grads --
    _grads_te = [vid_te0.grad,vid_te1.grad]
    _grads_gt = [vid_gt0.grad,vid_gt1.grad]
    for idx,(grads_te,grads_gt) in enumerate(zip(_grads_te,_grads_gt)):

        # -- viz --
        noedges = th.zeros_like(grads_te)
        H,W = noedges.shape[-2:]
        adj = (ps-1)//2+1
        noedges[...,adj:-adj,adj:-adj] = 1
        grads_te = noedges * grads_te
        grads_gt = noedges * grads_gt

        # -- skip l2 bwd --
        if dist_type == "l2": continue

        # -- compare grads --
        rel_error = th.abs(grads_gt - grads_te)/(th.abs(grads_gt)+1e-10)
        rel_error_nz = th.where(th.abs(grads_gt)>1e-3,rel_error,0.)

        # -- compare --
        tol = 1e-2
        error = th.max(rel_error_nz).item()
        if error > tol: print("Max Error: ",error)
        assert error < tol

        tol = 1e-5
        error = th.mean(rel_error_nz).item()
        if error > tol: print("Mean Error: ",error)
        # print("Mean Error: ",error)
        assert error < tol




