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
    vid /= vid.max()
    return vid

def pytest_generate_tests(metafunc):
    test_lists = {"wt":[0],"ws":[3],"k":[-1],"ps":[5],
                  "stride0":[2],"stride1":[1],"dilation":[1],
                  "nheads":[1],"anchor_self":[False],
                  "full_ws":[True],"dist_type":["l2"],
                  "seed":[0]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def test_fwd(ws,wt,k,ps,stride0,stride1,dilation,
             nheads,anchor_self,full_ws,dist_type,seed):
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
    reflect_bounds = False
    set_seed(seed)

    # -- load data --
    vid = get_data(dnames,ext)

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(run_flow,clean_flow,vid,vid,0.)
    flows.fflow = th.clamp(10*th.randn_like(flows.fflow),-10,10)
    flows.bflow = th.clamp(10*th.randn_like(flows.bflow),-10,10)

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
                                    full_ws=full_ws,anchor_self=anchor_self)
    search_gt = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                   dist_type=dist_type, dilation=dil,
                                   stride0=stride0, stride1=stride1,
                                   reflect_bounds=reflect_bounds,
                                   full_ws=full_ws,anchor_self=anchor_self,
                                   use_adj=False)

    # -- [testing] search --
    dists_te,inds_te = search_te(vid,vid,flows.fflow,flows.bflow)
    th.cuda.synchronize()

    # -- [groundtruth] search --
    dists_gt,inds_gt = search_gt(vid,vid,flows.fflow,flows.bflow)
    th.cuda.synchronize()

    # -- compare --
    args0 = th.where(th.logical_not(th.isinf(dists_gt))) # remove all inf
    diff = th.abs(dists_te - dists_gt) / (dists_gt.abs()+1e-5)
    args1 = th.where(diff > 1e-3)
    diff = diff[args0]

    # -- viz --
    # print(diff)
    # print(args1)
    # print(dists_te[args1])
    # print(dists_gt[args1])
    # print(inds_te[args1][:10])
    # print(inds_gt[args1][:10])

    # -- test --
    tol = 1e-5
    error = diff.mean().item()
    if error > tol: print("error: ",error)
    assert error < tol

    tol = 1e-4
    max_error = diff.max().item()
    if max_error > tol: print("max error: ",max_error)
    assert max_error < tol


def test_bwd(ws,wt,k,ps,stride0,stride1,dilation,
             nheads,anchor_self,full_ws,dist_type,seed):
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
    reflect_bounds = False
    adj = 0
    anchor_self = False
    set_seed(seed)

    # -- load data --
    vid = get_data(dnames,ext)
    vid = th.cat([vid,vid],-1)
    vid = th.cat([vid,vid],-2)

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(run_flow,clean_flow,vid,vid,0.)
    flows.fflow = th.clamp(10*th.randn_like(flows.fflow),-10,10)
    flows.bflow = th.clamp(10*th.randn_like(flows.bflow),-10,10)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape
    chnls = vid.shape[2]

    # -- allow grads --
    vid_te0,vid_te1 = vid.clone(),vid.clone()
    vid_te0.requires_grad_(True)
    vid_te1.requires_grad_(True)
    vid_gt0,vid_gt1 = vid.clone(),vid.clone()
    vid_gt0.requires_grad_(True)
    vid_gt1.requires_grad_(True)

    # -- exec fold fxns --
    sch = stnls.search
    search_te = sch.N3MatMultSearch(ws, wt, ps, k, nheads,
                                    dist_type=dist_type, dilation=dil,
                                    stride0=stride0, stride1=stride1,
                                    reflect_bounds=reflect_bounds,
                                    full_ws=full_ws,anchor_self=anchor_self)
    search_gt = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                   dist_type=dist_type, dilation=dil,
                                   stride0=stride0, stride1=stride1,
                                   reflect_bounds=reflect_bounds,
                                   full_ws=full_ws,anchor_self=anchor_self,
                                   use_adj=False,use_atomic=True)

    # -- [testing] search --
    dists_te,inds_te = search_te(vid_te0,vid_te1,flows.fflow,flows.bflow)
    th.cuda.synchronize()

    # -- [groundtruth] search --
    dists_gt,inds_gt = search_gt(vid_gt0,vid_gt1,flows.fflow,flows.bflow)
    th.cuda.synchronize()

    # -- compare --
    args0 = th.where(th.logical_not(th.isinf(dists_gt))) # remove all inf
    diff = th.abs(dists_te - dists_gt) / (dists_gt.abs()+1e-5)
    args1 = th.where(diff>1-3)

    tol = 1e-5
    error = diff[args0].mean().item()
    if error > tol: print("error: ",error)
    assert error < tol

    tol = 1e-4
    max_error = diff[args0].max().item()
    if max_error > tol: print("max error: ",max_error)
    assert max_error < tol

    # -- compute bwd --
    dists_grad = th.randn_like(dists_te)
    th.autograd.backward(dists_te,dists_grad)
    th.autograd.backward(dists_gt,dists_grad)

    # -- for both grads --
    _grads_te = [vid_te0.grad,vid_te1.grad]
    _grads_gt = [vid_gt0.grad,vid_gt1.grad]
    for idx,(grads_te,grads_gt) in enumerate(zip(_grads_te,_grads_gt)):

        # print(grads_te.shape)
        # print(grads_gt.shape)

        # -- viz [the error map may look weird] --
        # print("-"*20)
        # print(grads_te[0,0,0,:3,:3])
        # print(grads_gt[0,0,0,:3,:3])
        # print(grads_gt[0,0,0,:3,:3]/grads_te[0,0,0,:3,:3])
        # print("-"*20)
        # print(grads_te[0,0,-1,-3:,-3:])
        # print(grads_gt[0,0,-1,-3:,-3:])
        # print("-"*20)
        # print(grads_te[0,0,-3:,-3:])
        # print(grads_gt[0,0,-3:,-3:])
        # print("-"*20)
        # print(grads_te[0,0,10:13,10:13])
        # print(grads_gt[0,0,10:13,10:13])
        # print("-"*20)
        # print(grads_te[0,0,:3,:3])
        # print(grads_gt[0,0,:3,:3])
        # print("-"*20)

        # diff = (grads_te -grads_gt).abs()/(grads_gt.abs()+1e-8)
        # print(diff.max())
        # diff /= diff.max()
        # stnls.testing.data.save_burst(diff[:,[0]],SAVE_DIR,"grad_diff_0_%d" % exact)
        # stnls.testing.data.save_burst(diff[:,[1]],SAVE_DIR,"grad_diff_1_%d" % exact)
        # stnls.testing.data.save_burst(diff[:,[2]],SAVE_DIR,"grad_diff_2_%d" % exact)

        # -- compare grads --
        rel_error = th.abs(grads_gt - grads_te)/(th.abs(grads_gt)+1e-10)
        rel_error_nz = th.where(th.abs(grads_gt)>1e-3,rel_error,0.)

        tol = 1e-2
        error = th.max(rel_error_nz).item()
        if error > tol: print("Max Error: ",error)
        assert error < tol

        tol = 1e-5
        error = th.mean(rel_error_nz).item()
        if error > tol: print("Mean Error: ",error)
        # print("Mean Error: ",error)
        assert error < tol
