
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
    test_lists = {"ps":[7],"stride0":[4],"stride1":[1],
                  "dilation":[1],"wt":[0],"ws":[3],
                  "k":[-1],"exact":[False],"nheads":[1],
                  "anchor_self":[False],"seed":[0],"dist_type":["prod"],
                  "k_agg":[-1]}
    # test_lists = {"ps":[7,11],"stride0":[4],"stride1":[1,8],
    #               "dilation":[1,2],"wt":[2],"ws":[3,7],
    #               "k":[-1,10],"exact":[True],"nheads":[2],
    #               "seed":[0,1,3],"anchor_self":[False],"dist_type":["prod"]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)


def test_fwd_vs_int(ws,wt,k,ps,stride0,stride1,dilation,k_agg,
                    nheads,anchor_self,exact,dist_type,seed):
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
    wt = 1 if wt == 0 else wt # skip 0
    # wt = 0

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    reflect_bounds = True
    use_k = k > 0
    use_adj = False
    adj = 0
    off_H0,off_W0,off_H1,off_W1 = 0,0,0,0
    full_ws = True

    # -- load data --
    vid = get_data(dnames,ext)

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)
    # flows.fflow = 10*th.ones_like(flows.fflow)
    # flows.bflow = 10*th.ones_like(flows.bflow)
    # flows.fflow = th.clamp(10*th.randn_like(flows.fflow),-10,10)
    # flows.bflow = th.clamp(10*th.randn_like(flows.bflow),-10,10)
    flows.fflow = th.clamp(10*th.randn_like(flows.fflow),-10,10).round()
    flows.bflow = th.clamp(10*th.randn_like(flows.bflow),-10,10).round()
    # flows.fflow = th.clamp((10*th.randn_like(flows.fflow))//2,-10,10)
    # flows.bflow = th.clamp((10*th.randn_like(flows.bflow))//2,-10,10)
    # flows.fflow = th.clamp(10*th.randn_like(flows.fflow),-10,10)
    # flows.bflow = th.clamp(10*th.randn_like(flows.bflow),-10,10)
    # flows.fflow = 10*th.randn_like(flows.fflow)
    # flows.bflow = 10*th.randn_like(flows.bflow)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape


    # -- exec fold fxns --
    sch = stnls.search
    search_te = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                   dist_type=dist_type, dilation=dil,
                                   stride0=stride0, stride1=stride1,
                                   reflect_bounds=reflect_bounds,
                                   full_ws=full_ws,anchor_self=anchor_self,
                                   use_adj=use_adj,normalize_bwd=False,
                                   itype_fwd="float")
    search_gt = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                   dist_type=dist_type, dilation=dil,
                                   stride0=stride0, stride1=stride1,
                                   reflect_bounds=reflect_bounds,
                                   full_ws=full_ws,anchor_self=anchor_self,
                                   use_adj=use_adj,normalize_bwd=False,
                                   itype_fwd="int")

    # -- test api --
    # print(stnls.search.nls(vid,vid,flows.fflow,flows.bflow,
    #                                ws, wt, ps, k))

    # -- [testing] search --
    dists_te,inds_te = search_te(vid,vid,flows.fflow,flows.bflow)
    th.cuda.synchronize()

    # -- [groundtruth] search --
    dists_gt,inds_gt = search_gt(vid,vid,flows.fflow,flows.bflow)
    th.cuda.synchronize()

    # -- viz --
    # print(th.where(dists_te < -1000))
    # print(dists_te[th.where(dists_te < -1000)])
    # print(th.where(dists_gt < -1000))
    # print(dists_te[th.where(dists_gt < -1000)])
    print(flows.fflow[0,0,:,31,0])
    print(flows.fflow[0,0,:,0,31])
    # print(flows.fflow[0,0,:,31,0])
    print("-"*30)
    print(dists_te[1,0,188])
    print(dists_gt[1,0,188])
    print(inds_te[1,0,188])
    print(inds_gt[1,0,188])
    # print("-"*30)
    # print("-"*30)
    # print(dists_te[0,0,-1])
    # print(dists_gt[0,0,-1])
    # print(inds_te[0,0,-1])
    # print(inds_gt[0,0,-1])
    # print(inds_te[0,0,-1])
    # print(inds_gt[0,0,-1])
    # print(inds_te[0,0,-1] - inds_gt[0,0,-1])
    # print((dists_te/dists_gt)[0,0,0])
    # print(dists_te[0,0,-1])
    # print(dists_gt[0,0,-1])
    # print((dists_te/dists_gt)[0,0,-1])
    # print(inds_te[0,0,0,:])
    # print(inds_gt[0,0,0,:])
    # print(dists_te[0,0,15])
    # print(dists_gt[0,0,15])
    # print(inds_te[0,0,15])
    # print(inds_te[0,0,15]-inds_gt[0,0,15])
    # print(dists_te[0,0,0,:])
    # print(dists_gt[0,0,0,:])
    # print(inds_te[0,0,0,:])
    # print(inds_gt[0,0,0,:])
    # print(inds_te[0,0,256,:])
    # print(inds_gt[0,0,256,:])
    # print(dists_te[0,0,1,:10])
    # print(dists_gt[0,0,1,:10])
    # print(dists_te[0,0,17,:])
    # print(dists_gt[0,0,17,:])
    # print(dists_te.shape)
    # print(dists_gt.shape)

    # -- viz --
    # diff = th.abs(dists_te - dists_gt).mean((-1,-2))
    # if diff.max() > 1e-5: diff /= diff.max()
    # diff = repeat(diff,'h w -> 1 c h w',c=3)
    # stnls.testing.data.save_burst(diff,SAVE_DIR,"nn2_diff")

    # diff = th.abs(dists_te - dists_gt).mean((0,1))
    # if diff.max() > 1e-5: diff /= diff.max()
    # diff = repeat(diff,'h w -> 1 c h w',c=3)
    # stnls.testing.data.save_burst(diff,SAVE_DIR,"nn2_diff_t")

    # -- compare --
    args0 = th.where(th.logical_not(th.isinf(dists_gt))) # remove all inf
    diff = th.abs(dists_te - dists_gt) / (dists_gt.abs()+1e-5)
    args1 = th.where(diff > 1e-3)
    print(th.where(th.isnan(diff)))
    print(dists_te[th.where(th.isnan(diff))])
    print(dists_gt[th.where(th.isnan(diff))])
    diff = diff[args0]

    # -- viz --
    # print(diff)
    # print(args1)
    # print(dists_te[args1])
    # print(dists_gt[args1])
    # print(inds_te[args1][:10])
    # print(inds_gt[args1][:10])

    # -- test --
    tol = 5e-3
    error = diff.mean().item()
    if error > tol: print("error: ",error)
    assert error < tol

    tol = 1e-4
    max_error = diff.max().item()
    if max_error > tol: print("max error: ",max_error)
    assert max_error < tol


def test_bwd_vs_int(ws,wt,k,ps,stride0,stride1,k_agg,
                    dilation,nheads,exact,dist_type,seed):
    """

    Test the CUDA code with torch code

    Forward Pass

    """


    # -- get args --
    dil = dilation
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    pt = 1
    wt = 1 if wt == 0 else wt # skip 0

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    reflect_bounds = True
    use_k = k > 0
    use_adj = False
    anchor_self = False

    # -- load data --
    vid = get_data(dnames,ext)

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)
    flows.fflow = 10*th.zeros_like(flows.fflow)
    flows.bflow = 10*th.zeros_like(flows.bflow)
    # flows.fflow = 10*th.ones_like(flows.fflow)
    # flows.bflow = 10*th.ones_like(flows.bflow)
    # flows.fflow = 10*th.randn_like(flows.fflow)
    # flows.bflow = 10*th.randn_like(flows.bflow)

    # -- allow grads --
    vid_te0,vid_te1 = vid.clone(),vid.clone()
    vid_te0.requires_grad_(True)
    vid_te1.requires_grad_(True)
    vid_gt0,vid_gt1 = vid.clone(),vid.clone()
    vid_gt0.requires_grad_(True)
    vid_gt1.requires_grad_(True)

    # -- exec fold fxns --
    sch = stnls.search
    search_te = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                   dist_type=dist_type,
                                   dilation=dil,stride0=stride0, stride1=stride1,
                                   reflect_bounds=reflect_bounds,
                                   full_ws=False,full_ws_time=False,
                                   use_adj=use_adj,anchor_self=anchor_self,
                                   itype_fwd="float",itype_bwd="float")
    search_gt = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                   dist_type=dist_type,
                                   dilation=dil,stride0=stride0, stride1=stride1,
                                   reflect_bounds=reflect_bounds,
                                   full_ws=False,full_ws_time=False,
                                   use_adj=use_adj,anchor_self=anchor_self,
                                   itype_fwd="int",itype_bwd="int")

    # -- [testing] search --
    # print("vid.shape: ",vid.shape)
    dists_te,inds_te = search_te(vid_te0,vid_te1,flows.fflow,flows.bflow)
    th.cuda.synchronize()

    # -- [groundtruth] search --
    dists_gt,inds_gt = search_gt(vid_gt0,vid_gt1,flows.fflow,flows.bflow)
    th.cuda.synchronize()
    # th.autograd.gradcheck

    # -- viz --
    # print(dists_te)
    # print(dists_gt)
    # print(dists_te[0,0,0])
    # print(dists_gt[0,0,0])
    # print(inds_te[0,0,0])
    # print(inds_gt[0,0,0])
    # print(dists_te.shape)
    # print(dists_gt.shape)

    # -- viz --
    # diff = th.abs(dists_te - dists_gt).mean((-1,-2))
    # if diff.max() > 1e-5: diff /= diff.max()
    # diff = repeat(diff,'h w -> 1 c h w',c=3)
    # stnls.testing.data.save_burst(diff,SAVE_DIR,"nn2_diff")

    # diff = th.abs(dists_te - dists_gt).mean((0,1))
    # if diff.max() > 1e-5: diff /= diff.max()
    # diff = repeat(diff,'h w -> 1 c h w',c=3)
    # stnls.testing.data.save_burst(diff,SAVE_DIR,"nn2_diff_t")

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

    # -- view --
    # print(vid_te0.grad[0,0,:3,:3])
    # print(vid_te1.grad[0,0,:3,:3])

    # -- for both grads --
    _grads_te = [vid_te0.grad,vid_te1.grad]
    _grads_gt = [vid_gt0.grad,vid_gt1.grad]
    for idx,(grads_te,grads_gt) in enumerate(zip(_grads_te,_grads_gt)):

        # # -- viz --
        # print(grads_gt[0,0,0,:3,:3])
        # print(grads_te[0,0,0,:3,:3])
        # print((grads_gt/grads_te)[0,0,0,:3,:3])
        # print("-"*30)
        # print(grads_gt[0,0,0,-3:,-3:])
        # print(grads_te[0,0,0,-3:,-3:])
        # print((grads_gt/grads_te)[0,0,0,-3:,-3:])

        # -- viz [the error map may look weird] --
        # print(grads_te.shape,grads_gt.shape)
        # print("-"*20)
        # print(grads_te[0,0,-1,:10,:10])
        # print(grads_gt[0,0,-1,:10,:10])
        # print((grads_te/grads_gt)[0,0,-1,:10,:10])
        # print("-"*20)
        # print(grads_te[0,0,-1,-10:,-10:])
        # print(grads_gt[0,0,-1,-10:,-10:])
        # print((grads_te/grads_gt)[0,0,-1,-10:,-10:])
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

        # diff = th.abs(grads_te -grads_gt)/(th.abs(grads_gt)+1e-10)
        # print(diff.max())
        # diff /= diff.max()
        # stnls.testing.data.save_burst(diff[:,[0],0],SAVE_DIR,"grad_diff_0_%d" % exact)
        # stnls.testing.data.save_burst(diff[:,[1],0],SAVE_DIR,"grad_diff_1_%d" % exact)
        # stnls.testing.data.save_burst(diff[:,[2],0],SAVE_DIR,"grad_diff_2_%d" % exact)

        # -- compare grads --
        rel_error = th.abs(grads_gt - grads_te)/(th.abs(grads_gt)+1e-10)
        rel_error_nz  = rel_error
        args = th.where(th.abs(grads_gt)>1e-3)

        tol = 1e-2
        error = th.max(rel_error_nz[args]).item()
        if error > tol: print("Max Error: ",error)
        # print("Max Error: ",error)
        assert error < tol

        tol = 1e-3
        error = th.mean(rel_error_nz[args]).item()
        if error > tol: print("Mean Error: ",error)
        # print("Mean Error: ",error)
        assert error < tol


def test_bwd_flows(ws,wt,k,ps,stride0,stride1,k_agg,
                   dilation,nheads,exact,dist_type,seed):
    """

    Test the CUDA code with torch code

    Forward Pass

    """


    # -- get args --
    dil = dilation
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    pt = 1
    wt = 1 if wt == 0 else wt # skip 0

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    reflect_bounds = True
    use_k = k > 0
    use_adj = False
    anchor_self = False

    # -- load data --
    vid = get_data(dnames,ext)

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)
    # flows.fflow = 10*th.zeros_like(flows.fflow)
    # flows.bflow = 10*th.zeros_like(flows.bflow)
    # flows.fflow = 10*th.ones_like(flows.fflow)
    # flows.bflow = 10*th.ones_like(flows.bflow)
    flows.fflow = 10*th.randn_like(flows.fflow)
    flows.bflow = 10*th.randn_like(flows.bflow)

    # -- allow grads --
    vid_te0,vid_te1 = vid.clone(),vid.clone()
    fflow_te = flows.fflow.clone()
    bflow_te = flows.bflow.clone()
    fflow_te.requires_grad_(True)
    bflow_te.requires_grad_(True)
    vid_te0.requires_grad_(True)
    vid_te1.requires_grad_(True)
    vid_gt0,vid_gt1 = vid.clone(),vid.clone()
    fflow_gt = flows.fflow.clone()
    bflow_gt = flows.bflow.clone()
    fflow_gt.requires_grad_(True)
    bflow_gt.requires_grad_(True)
    vid_gt0.requires_grad_(True)
    vid_gt1.requires_grad_(True)

    # -- exec fold fxns --
    sch = stnls.search
    search_te = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                   dist_type=dist_type,
                                   dilation=dil,stride0=stride0, stride1=stride1,
                                   reflect_bounds=reflect_bounds,
                                   full_ws=False,full_ws_time=False,
                                   use_adj=use_adj,anchor_self=anchor_self,
                                   itype_fwd="float",itype_bwd="float")
    search_gt = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                   dist_type=dist_type,
                                   dilation=dil,stride0=stride0, stride1=stride1,
                                   reflect_bounds=reflect_bounds,
                                   full_ws=False,full_ws_time=False,
                                   use_adj=use_adj,anchor_self=anchor_self,
                                   itype_fwd="int",itype_bwd="int")

    # -- [testing] search --
    # print("vid.shape: ",vid.shape)
    dists_te,inds_te = search_te(vid_te0,vid_te1,fflow_te,bflow_te)
    th.cuda.synchronize()

    # -- [groundtruth] search --
    dists_gt,inds_gt = search_gt(vid_gt0,vid_gt1,flows.fflow,flows.bflow)
    th.cuda.synchronize()
    # th.autograd.gradcheck

    # -- viz --
    # print(dists_te)
    # print(dists_gt)
    # print(dists_te[0,0,0])
    # print(dists_gt[0,0,0])
    # print(inds_te[0,0,0])
    # print(inds_gt[0,0,0])
    # print(dists_te.shape)
    # print(dists_gt.shape)

    # -- viz --
    # diff = th.abs(dists_te - dists_gt).mean((-1,-2))
    # if diff.max() > 1e-5: diff /= diff.max()
    # diff = repeat(diff,'h w -> 1 c h w',c=3)
    # stnls.testing.data.save_burst(diff,SAVE_DIR,"nn2_diff")

    # diff = th.abs(dists_te - dists_gt).mean((0,1))
    # if diff.max() > 1e-5: diff /= diff.max()
    # diff = repeat(diff,'h w -> 1 c h w',c=3)
    # stnls.testing.data.save_burst(diff,SAVE_DIR,"nn2_diff_t")

    # -- compare --
    # args0 = th.where(th.logical_not(th.isinf(dists_gt))) # remove all inf
    # diff = th.abs(dists_te - dists_gt) / (dists_gt.abs()+1e-5)
    # args1 = th.where(diff>1-3)

    # tol = 1e-5
    # error = diff[args0].mean().item()
    # if error > tol: print("error: ",error)
    # assert error < tol

    # tol = 1e-4
    # max_error = diff[args0].max().item()
    # if max_error > tol: print("max error: ",max_error)
    # assert max_error < tol

    # -- compute bwd --
    # grad_dists = th.randn_like(dists_te)
    # th.autograd.backward(dists_te,grad_dists)
    # th.autograd.backward(dists_gt,grad_dists)
    grad_inds = th.randn_like(inds_te)
    th.autograd.backward(inds_te,grad_inds)

    # -- view --
    # print(vid_te0.grad[0,0,:3,:3])
    # print(vid_te1.grad[0,0,:3,:3])

    # -- for both grads --
    _grads_te = [fflow_te.grad,bflow_te.grad]
    _grads_gt = [fflow_gt.grad,bflow_gt.grad]
    _grads_gt = [None,None]
    for idx,(grads_te,grads_gt) in enumerate(zip(_grads_te,_grads_gt)):

        print(grads_te)
        exit()
        # # -- viz --
        # print(grads_gt[0,0,0,:3,:3])
        # print(grads_te[0,0,0,:3,:3])
        # print((grads_gt/grads_te)[0,0,0,:3,:3])
        # print("-"*30)
        # print(grads_gt[0,0,0,-3:,-3:])
        # print(grads_te[0,0,0,-3:,-3:])
        # print((grads_gt/grads_te)[0,0,0,-3:,-3:])

        # -- viz [the error map may look weird] --
        # print(grads_te.shape,grads_gt.shape)
        # print("-"*20)
        # print(grads_te[0,0,-1,:10,:10])
        # print(grads_gt[0,0,-1,:10,:10])
        # print((grads_te/grads_gt)[0,0,-1,:10,:10])
        # print("-"*20)
        # print(grads_te[0,0,-1,-10:,-10:])
        # print(grads_gt[0,0,-1,-10:,-10:])
        # print((grads_te/grads_gt)[0,0,-1,-10:,-10:])
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

        # diff = th.abs(grads_te -grads_gt)/(th.abs(grads_gt)+1e-10)
        # print(diff.max())
        # diff /= diff.max()
        # stnls.testing.data.save_burst(diff[:,[0],0],SAVE_DIR,"grad_diff_0_%d" % exact)
        # stnls.testing.data.save_burst(diff[:,[1],0],SAVE_DIR,"grad_diff_1_%d" % exact)
        # stnls.testing.data.save_burst(diff[:,[2],0],SAVE_DIR,"grad_diff_2_%d" % exact)

        # -- compare grads --
        rel_error = th.abs(grads_gt - grads_te)/(th.abs(grads_gt)+1e-10)
        rel_error_nz  = rel_error
        args = th.where(th.abs(grads_gt)>1e-3)

        tol = 1e-2
        error = th.max(rel_error_nz[args]).item()
        if error > tol: print("Max Error: ",error)
        # print("Max Error: ",error)
        assert error < tol

        tol = 1e-3
        error = th.mean(rel_error_nz[args]).item()
        if error > tol: print("Mean Error: ",error)
        # print("Mean Error: ",error)
        assert error < tol


