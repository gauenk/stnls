
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
                  "dilation":[1],"wt":[1],"ws":[3],
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

# @pytest.mark.skip
def test_fwd(ws,wt,k,ps,stride0,stride1,dilation,k_agg,
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
    seed = 234
    set_seed(seed)

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    reflect_bounds = True
    use_k = k > 0
    use_adj = False
    adj = 0
    off_H0,off_W0,off_H1,off_W1 = 0,0,0,0

    # -- load data --
    vid = get_data(dnames,ext)
    # vid = th.ones_like(vid)

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)
    # flows.fflow = 10*th.ones_like(flows.fflow)
    # flows.bflow = 10*th.ones_like(flows.bflow)
    flows.fflow = th.clamp(3*th.randn_like(flows.fflow)**2,-10,10)
    flows.bflow = th.clamp(3*th.randn_like(flows.bflow)**2,-10,10)

    # flows.fflow = th.clamp(10*th.randn_like(flows.fflow),-10,10).round()
    # flows.bflow = th.clamp(10*th.randn_like(flows.bflow),-10,10).round()

    # Z = 0.75
    # Z = 0.75
    # flows.fflow = th.clamp(Z*th.ones_like(flows.fflow),-10,10)
    # flows.bflow = th.clamp(Z*th.ones_like(flows.bflow),-10,10)
    # flows.fflow = th.clamp(10*th.zeros_like(flows.fflow),-10,10)
    # flows.bflow = th.clamp(10*th.zeros_like(flows.bflow),-10,10)

    # flows.fflow = th.clamp((10*th.randn_like(flows.fflow))//2,-10,10)
    # flows.bflow = th.clamp((10*th.randn_like(flows.bflow))//2,-10,10)
    # flows.fflow = th.clamp(10*th.randn_like(flows.fflow),-10,10)
    # flows.bflow = th.clamp(10*th.randn_like(flows.bflow),-10,10)
    # flows.fflow = 10*th.randn_like(flows.fflow)
    # flows.bflow = 10*th.randn_like(flows.bflow)

    # -- unpack image --
    # flows.fflow = flows.fflow.float()
    # flows.bflow = flows.bflow.float()
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape
    itype_fwd = "float"
    itype_bwd = "float"

    # -- exec fold fxns --
    sch = stnls.search
    search_te = sch.PairedSearch(ws, ps, k, nheads, dist_type=dist_type,
                                 dilation=dil,stride0=stride0, stride1=stride1,
                                 reflect_bounds=reflect_bounds,
                                 full_ws=True,full_ws_time=True,
                                 anchor_self=anchor_self,use_adj=use_adj,
                                 itype_fwd=itype_fwd,itype_bwd=itype_bwd)
    search_gt = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                   dist_type=dist_type,
                                   dilation=dil,stride0=stride0, stride1=stride1,
                                   reflect_bounds=reflect_bounds,
                                   full_ws=True,full_ws_time=True,
                                   anchor_self=anchor_self,use_adj=use_adj,
                                   itype_fwd=itype_fwd,itype_bwd=itype_bwd)

    # -- [groundtruth] search --
    dists_gt,inds_gt = search_gt(vid,vid,flows.fflow,flows.bflow)
    th.cuda.synchronize()
    print(dists_gt.shape,inds_gt.shape)

    # -- [testing] search --
    dtype = th.float if itype_fwd == "float" else th.int
    acc_flows = stnls.nn.accumulate_flow(flows,dtype=dtype)
    wt = 1
    dists_te,inds_te = [],[]
    zflow = th.zeros_like(acc_flows.fflow[:,0,0])
    B,T,_,H,W = vid.shape
    for ti in range(T):
        # if ti != 1: continue

        swap = False
        t_inc = 0
        prev_t = ti
        t_shift = min(0,ti-wt) + max(0,ti + wt - (T-1))
        t_max = min(T-1,ti + wt - t_shift);
        # print(t_shift,t_max)
        tj = ti

        dists_te_i,inds_te_i = [],[]
        for _tj in range(2*wt+1):

            # -- update search frame --
            prev_t = tj
            tj = prev_t + t_inc
            swap = tj > t_max
            t_inc = 1 if (t_inc == 0) else t_inc
            t_inc = -1 if swap else t_inc
            tj = ti-1 if swap else tj
            prev_t = ti if swap else prev_t
            # print(ti,tj,t_inc,swap)

            frame0 = vid[:,ti]
            frame1 = vid[:,tj]
            if ti == tj:
                flow = zflow
            elif ti < tj:
                # flow = acc_flows.fflow[:,tj - ti - 1]
                print("fwd: ",ti,tj,tj-ti-1)
                flow = acc_flows.fflow[:,ti,tj-ti-1]
            elif ti > tj:
                print("bwd: ",ti,tj,ti-tj-1)
                # flow = acc_flows.bflow[:,ti - tj - 1]
                flow = acc_flows.bflow[:,ti,ti-tj-1]
            flow = flow.float()
            if ti == 0:
                if tj == 1:
                    print(flow[0])
                    flow = flows.fflow[:,ti]
                    print(flow[0])
                    # print("hi.")
                # print(flow)
                # print(th.mean(flow-flows.fflow[:,ti]).item())
            dists_ij,inds_ij = search_te(frame0,frame1,flow)
            inds_t = tj*th.ones_like(inds_ij[...,[0]])
            inds_ij = th.cat([inds_t,inds_ij],-1)
            dists_te_i.append(dists_ij)
            inds_te_i.append(inds_ij)
        dists_te_i = th.cat(dists_te_i,-1)
        inds_te_i = th.cat(inds_te_i,-2)
        dists_te.append(dists_te_i)
        inds_te.append(inds_te_i)
    dists_te = th.cat(dists_te,-2)
    inds_te = th.cat(inds_te,-3)
    # return
    # dists_te = dists_te[:,:,256:512]
    # dists_gt = dists_gt[:,:,256:512]
    # print(dists_te)
    # print(dists_gt)
    # print(inds_te)
    # args = th.where(th.abs(dists_te - dists_gt) > 1e-2)
    # print(args)
    # print(dists_te[args])
    # print(dists_gt[args])

    # print(inds_gt.shape,inds_te.shape)
    # print(inds_gt[0,0,0,9:11])
    # print(inds_te[0,0,0,9:11])

    # print(inds_gt[0,0,0,:20])
    # print(inds_te[0,0,0,:20])
    # print(inds_gt[0,0,-1,:20])
    # print(inds_te[0,0,-1,:20])


    # -- select --
    # print(dists_te.shape)
    # dists_te = dists_te[:,:,256:512]
    # dists_gt = dists_gt[:,:,256:512]
    # dists_te = dists_te[...,9:10]
    # dists_gt = dists_gt[...,9:10]
    # combo = th.cat([dists_te,dists_gt],-1)[0,0]
    # print(combo.shape)
    # for i in range(256):
    #     print(i,combo[i],th.mean(combo[i][0] - combo[i][1]).item())

    # -- viz --
    # print(dists_te[0,0,-1])
    # print(dists_gt[0,0,-1])
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

def test_bwd(ws,wt,k,ps,stride0,stride1,k_agg,
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
                                   use_adj=use_adj,anchor_self=anchor_self)
    search_gt = stnls.search_dev.init("%s_search_with_heads" % dist_type,
                                      flows.fflow, flows.bflow,
                                      k, ps, pt, ws, wt, nheads,
                                      chnls=-1,dilation=dil,
                                      stride0=stride0, stride1=stride1,
                                      reflect_bounds=reflect_bounds,
                                      use_k=use_k,use_adj=use_adj,
                                      anchor_self=anchor_self,exact=True)


    # -- [testing] search --
    # print("vid.shape: ",vid.shape)
    dists_te,inds_te = search_te(vid_te0,vid_te1,flows.fflow,flows.bflow)
    th.cuda.synchronize()

    # -- [groundtruth] search --
    dists_gt,inds_gt = search_gt(vid_gt0,vid_gt1)
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



