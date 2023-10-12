
# -- python --
import sys
from dev_basics.utils import vid_io

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
                  "k":[-1],"nheads":[1],
                  "self_action":[None],"seed":[0],
                  "dist_type":["prod"],
                  "k_agg":[-1],}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)


def test_fwd_vs_int(ws,wt,k,ps,stride0,stride1,dilation,k_agg,
                    nheads,self_action,dist_type,seed):
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
                                   full_ws=full_ws,use_adj=use_adj,
                                   self_action=self_action,
                                   normalize_bwd=False,itype="float")
    search_gt = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                   dist_type=dist_type, dilation=dil,
                                   stride0=stride0, stride1=stride1,
                                   reflect_bounds=reflect_bounds,
                                   full_ws=full_ws,use_adj=use_adj,
                                   self_action=self_action,
                                   normalize_bwd=False,itype="int")

    # -- test api --
    # print(stnls.search.nls(vid,vid,flows.fflow,flows.bflow,
    #                                ws, wt, ps, k))

    # -- [testing] search --
    dists_te,inds_te = search_te(vid,vid,flows.fflow,flows.bflow)
    th.cuda.synchronize()

    # -- [groundtruth] search --
    dists_gt,inds_gt = search_gt(vid,vid,flows.fflow,flows.bflow)
    th.cuda.synchronize()

    # print(th.stack([dists_te.reshape(-1),dists_gt.reshape(-1)],-1))

    # -- viz --
    # print(th.where(dists_te < -1000))
    # print(dists_te[th.where(dists_te < -1000)])
    # print(th.where(dists_gt < -1000))
    # print(dists_te[th.where(dists_gt < -1000)])
    # print(flows.fflow[0,0,:,31,0])
    # print(flows.fflow[0,0,:,0,31])
    # # print(flows.fflow[0,0,:,31,0])
    # print("-"*30)
    # print(dists_te[1,0,188])
    # print(dists_gt[1,0,188])
    # print(inds_te[1,0,188])
    # print(inds_gt[1,0,188])
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
    args0 = th.where(th.logical_not((dists_gt.abs()>1e30))) # remove all inf
    diff = th.abs(dists_te - dists_gt) / (dists_gt.abs()+1e-5)
    args1 = th.where(diff > 1e-3)
    print(th.where(th.isnan(diff)))
    print(dists_te[th.where(th.isnan(diff))])
    print(dists_gt[th.where(th.isnan(diff))])
    inds = []
    for i in range(3):
        inds.append(inds_gt[...,i][th.where(th.isnan(diff))])
    inds = th.stack(inds,-1)
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


def test_bwd_vid(ws,wt,k,ps,stride0,stride1,k_agg,
                 dilation,nheads,self_action,dist_type,seed):
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
    full_ws = True

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
                                   full_ws=full_ws,use_adj=use_adj,
                                   self_action=self_action,itype="float")
    search_gt = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                   dist_type=dist_type,
                                   dilation=dil,stride0=stride0, stride1=stride1,
                                   reflect_bounds=reflect_bounds,
                                   full_ws=full_ws,use_adj=use_adj,
                                   self_action=self_action,itype="int")

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
    # print(vid_gt0.grad[0,0,:3,:3])
    # print(vid_te0.grad[0,0,:3,:3])
    # print(vid_te1.grad[0,0,:3,:3])

    # -- for both grads --
    _grads_te = [vid_te0.grad,vid_te1.grad]
    _grads_gt = [vid_gt0.grad,vid_gt1.grad]
    for idx,(grads_te,grads_gt) in enumerate(zip(_grads_te,_grads_gt)):

        # -- viz --
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


    # -- prep gradient check --
    # print("Running grad check.")
    vid = vid[...,:3,::2,::2]
    B,T,F,H,W = vid.shape
    vid0 = th.rand_like(vid).requires_grad_(True)
    vid1 = th.rand_like(vid).requires_grad_(True)
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    W_t = 2*wt
    flows = th.ones((B,1,T,W_t,2,nH,nW)).cuda()/2.
    flows = th.rand_like(flows)/2.+0.1 # away from ints

    # -- run gradient check
    search_gt_vid0 = lambda vid0: search_gt(vid0,vid1,flows)[0]
    th.autograd.gradcheck(search_gt_vid0, vid0, eps=1e-4,
                          atol=1e-2, nondet_tol=1e-7, raise_exception=True)
    search_gt_vid1 = lambda vid1: search_gt(vid0,vid1,flows)[0]
    th.autograd.gradcheck(search_gt_vid1, vid1, eps=1e-4,
                          atol=1e-2, nondet_tol=1e-7, raise_exception=True)


def test_bwd_flows(ws,wt,k,ps,stride0,stride1,k_agg,
                   dilation,nheads,self_action,dist_type,seed):
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
    set_seed(seed)

    # -- init vars --
    device = "cuda:0"
    reflect_bounds = False
    use_adj = False
    full_ws = True

    # -- load video --
    vid = get_data(dnames,ext)[...,:1,::2,::2]
    vid0,vid1 = vid.clone(),vid.flip(-1).clone()
    vid0 = th.rand_like(vid)-0.5
    vid1 = th.rand_like(vid)-0.5
    # vid0 = th.ones_like(vid)
    # vid0 = th.zeros_like(vid)
    # vid1 = th.ones_like(vid)
    # vid1 = th.zeros_like(vid)
    vid0 = vid0.round(decimals=3)
    vid1 = vid1.round(decimals=3)
    B,T,F,H,W = vid0.shape

    # -- load flows --
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    W_t = 2*wt
    flows = th.ones((B,1,T,W_t,2,nH,nW)).cuda()/2.
    flows = th.rand_like(flows)-0.5
    # flows = th.rand_like(flows)/2.+0.2 # away from ints
    # flows = -flows.round(decimals=4)
    # flows = th.rand_like(flows)/2.+4.2 # away from ints
    # print(th.any(flows.abs()<1e-3))
    # print("flows[min,max]: ",flows.min().item(),flows.max().item())
    flows.requires_grad_(True)

    # -- exec fold fxns --
    sch = stnls.search
    search_gt = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                   dist_type=dist_type,
                                   dilation=dil,stride0=stride0, stride1=stride1,
                                   reflect_bounds=reflect_bounds,
                                   full_ws=full_ws,use_adj=use_adj,
                                   self_action=self_action,itype="float")

    # -- gradient check --
    search_gt_flows = lambda flows: search_gt(vid0,vid1,flows)[0]
    th.autograd.gradcheck(search_gt_flows, flows, eps=1e-3,
                          atol=1e-2, nondet_tol=1e-7, raise_exception=True)

    search_gt_flows = lambda flows: search_gt(vid0,vid1,flows)[1]
    th.autograd.gradcheck(search_gt_flows, flows, eps=1e-3,
                          atol=1e-2, nondet_tol=1e-7, raise_exception=True)

    # # -- gradient check --
    # from torch.autograd.gradcheck import get_numerical_jacobian,get_analytical_jacobian
    # from torch.autograd.gradcheck import _get_numerical_jacobian
    # from torch.autograd.gradcheck import _check_analytical_jacobian_attributes
    # num = _get_numerical_jacobian(search_gt_flows, (flows,),
    #                               eps=1e-3, is_forward_ad=False)[0][0]
    # out = search_gt_flows(flows)
    # ana = _check_analytical_jacobian_attributes((flows,), out, 1e-7, False)[0]
    # print(num.shape,ana.shape)

    # args = th.where(num.abs()>0)
    # print(num[args][:10])
    # print(ana[args][:10])

    # diff = th.abs(num - ana)
    # print(th.mean(diff))
    # print(th.max(diff))
    # print(th.min(diff))
    # print(th.sum(1.*(diff > 1e-2)))
    # print(th.where(diff > 1e-2))
    # print(num[th.where(diff > 1e-2)][100:110])
    # print(ana[th.where(diff > 1e-2)][100:110])
    # print(th.all(num[th.where(diff > 1e-2)] == 0))
    # # for i in range(100):
    # #     print("Num NZ @ row0: ",
    # #           th.sum(1.*(num[i].abs() > 0)).item(),
    # #           th.sum(1.*(ana[i].abs() > 0)).item())
    # # #     print("Num NZ @ col0: ",th.sum(1.*(num[:,i].abs() > 0)))
    # # print("[in/out]: ",flows.numel(),out.numel())


def test_fwd_anchor(ws,wt,ps,stride0,stride1,dilation,
                    self_action,dist_type,seed):

    """

    Test the CUDA code with torch code

    Forward Pass

    """

    # -- init vars --
    dil = dilation
    pt = 1
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    reflect_bounds = False
    use_adj = False
    full_ws = True
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    topk_mode = "each"
    itype = "float"
    nheads = 1

    # -- load data --
    vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid.to(device)[:,:5,:3,::2,::2].contiguous()
    vid = repeat(vid,'b t c h w -> b t (r c) h w',r=12)[:,:32].contiguous()
    vid /= vid.max()
    vid0 = th.rand_like(vid)-0.5
    vid1 = th.rand_like(vid)-0.5

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)
    fflow = 10*th.randn_like(flows.fflow)
    bflow = 10*th.randn_like(flows.bflow)
    B,T,F,H,W = vid.shape
    W_t = 2*wt+1
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    flows = 2*th.rand((B,T,W_t-1,2,nH,nW)).to(vid0.device)

    # -- exec fold fxns --
    k0 = ws*ws
    search0 = stnls.search.NonLocalSearch(ws, wt, ps, -1, nheads,
                                          dilation=dil,stride0=stride0, stride1=stride1,
                                          reflect_bounds=reflect_bounds,full_ws=False,
                                          self_action=None,use_adj=use_adj,
                                          dist_type=dist_type,topk_mode=topk_mode,
                                          itype=itype)
    k1 = 3
    search1 = stnls.search.NonLocalSearch(ws, wt, ps, k1, nheads,
                                          dilation=dil,stride0=stride0, stride1=stride1,
                                          reflect_bounds=reflect_bounds,full_ws=False,
                                          self_action="anchor_each",use_adj=use_adj,
                                          dist_type=dist_type,topk_mode=topk_mode,
                                          itype=itype)
    k2 = 5
    search2 = stnls.search.NonLocalSearch(ws, wt, ps, k2, nheads,
                                          dilation=dil,stride0=stride0, stride1=stride1,
                                          reflect_bounds=reflect_bounds,full_ws=True,
                                          self_action="anchor_each",use_adj=use_adj,
                                          dist_type=dist_type,topk_mode=topk_mode,
                                          itype=itype)



    # -- exec --
    HD = nheads
    vshape = (B,HD,T,nH,nW,W_t)

    dists0,inds0 = search0(vid0,vid1,flows)
    dists0,inds0 = dists0.view(vshape+(k0,)),inds0.view(vshape+(k0,3,))
    dists0 = dists0[...,:,ws//2+ws]
    inds0= inds0[...,:,ws//2+ws,:]

    dists1,inds1 = search1(vid0,vid1,flows)
    dists1,inds1 = dists1.view(vshape+(k1,)),inds1.view(vshape+(k1,3,))
    dists1 = dists1[...,:,0]
    inds1= inds1[...,:,0,:]


    dists2,inds2 = search2(vid0,vid1,flows)
    dists2,inds2 = dists2.view(vshape+(k2,)),inds2.view(vshape+(k2,3,))
    dists2 = dists2[...,:,0]
    inds2= inds2[...,:,0,:]


    # -- check all pairwise --
    dists = [dists0,dists1,dists2]
    inds = [inds0,inds1,inds2]
    for i in range(3):
        for j in range(3):
            if i == j: continue
            assert th.allclose(dists[i],dists[j],1e-3,1e-3,equal_nan=True)
            assert th.allclose(inds[i],inds[j],1e-3,1e-3,equal_nan=True)

    # -- check against flow --
    def reflect_bounds(flow,i,L):
        args = th.where(flow[...,i] >= L)
        flow[...,i][args] = 2*L - flow[...,i][args]
        args = th.where(flow[...,i] < 0)
        flow[...,i][args] = -flow[...,i][args]
    grid = stnls.nn.index_grid(nH,nW).flip(1)*stride0
    for i in range(3):
        inds_i = inds[i]
        for ti in range(T):
            for si in range(W_t):
                ind = inds_i[:,0,ti,:,:,si,1:]
                if si > 0:
                    flow = flows[:,ti,si-1].flip(1) + grid
                else:
                    flow = th.zeros_like(flows[:,ti,0]).flip(1) + grid
                flow = rearrange(flow,'b i h w -> b h w i')
                reflect_bounds(flow,0,H)
                reflect_bounds(flow,1,W)
                diff = th.mean(th.abs(ind - flow)).item()
                assert th.allclose(ind,flow,1e-3,1e-3,equal_nan=True)
