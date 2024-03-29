
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
    test_lists = {"ps":[3],"stride0":[1],
                  "stride1":[1.1],
                  "dilation":[1],"wt":[1],"ws":[3],
                  "k":[-1],"nheads":[2],
                  "self_action":[None],"seed":[0],
                  "dist_type":["l2","prod"],
                  "reflect_bounds":[True],
                  "itype":["float","int"]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)


def test_fwd_vs_int(ws,wt,k,ps,stride0,stride1,dilation,reflect_bounds,
                    nheads,self_action,dist_type,seed,itype):
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
    full_ws = True
    stride1 = int(stride1) # no fractional stride1 for this test

    # -- load data --
    B,T,HD,F,H,W = 1,3,nheads,1,10,10
    vid = th.ones((B,T,HD*F,H,W),device=device)
    vid = th.rand_like(vid)

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)
    flows.fflow = th.clamp(10*th.randn_like(flows.fflow),-10,10).round()
    flows.bflow = th.clamp(10*th.randn_like(flows.bflow),-10,10).round()
    # flows = stnls.nn.search_flows(flows.fflow,flows.bflow,wt,stride0)

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
                                   full_ws=full_ws,
                                   self_action=self_action,
                                   normalize_bwd=False,itype="float")
    search_gt = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                   dist_type=dist_type, dilation=dil,
                                   stride0=stride0, stride1=stride1,
                                   reflect_bounds=reflect_bounds,
                                   full_ws=full_ws,
                                   self_action=self_action,
                                   normalize_bwd=False,itype="int")

    # -- [testing] search --
    dists_te,inds_te = search_te(vid,vid,flows.fflow,flows.bflow)
    th.cuda.synchronize()

    # -- [groundtruth] search --
    dists_gt,inds_gt = search_gt(vid,vid,flows.fflow,flows.bflow)
    th.cuda.synchronize()

    # -- compare --
    args0 = th.where(th.logical_not((dists_gt.abs()>1e30))) # remove all inf
    diff = th.abs(dists_te - dists_gt) / (dists_gt.abs()+1e-5)
    diff = diff[args0]

    # -- test --
    tol = 5e-3
    error = diff.mean().item()
    if error > tol: print("error: ",error)
    assert error < tol

    tol = 1e-4
    max_error = diff.max().item()
    if max_error > tol: print("max error: ",max_error)
    assert max_error < tol


def test_bwd_vid_int_bilin2d(ws,wt,k,ps,stride0,stride1,
                             dilation,nheads,self_action,dist_type,seed):
    """

    Test the CUDA code with torch code

    Forward Pass

    """


    # -- get args --
    dil = dilation
    pt = 1
    wt = 1 if wt == 0 else wt # skip 0
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    reflect_bounds = True
    use_adj = False
    full_ws = True
    stride1 = int(stride1) # no fraction for this test
    set_seed(seed)

    # -- load data --
    B,T,HD,F,H,W = 1,3,nheads,1,10,10
    vid = th.ones((B,T,HD*F,H,W),device=device)
    vid = th.rand_like(vid)
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)
    flows.fflow = flows.fflow.round()
    flows.bflow = flows.bflow.round()

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
        assert th.allclose(grads_te,grads_gt,1e-3,1e-3,equal_nan=True)


def test_bwd_vid_flowgrad_noflowgrad(ws,wt,k,ps,stride0,stride1,reflect_bounds,
                                     dilation,nheads,self_action,dist_type,seed):
    """

    Test the CUDA code with torch code

    Forward Pass

    """


    # -- get args --
    dil = dilation
    ext = "jpg"
    pt = 1
    wt = 1 if wt == 0 else wt # skip 0
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    use_adj = False
    full_ws = True
    set_seed(seed)

    # -- load data --
    B,T,HD,F,H,W = 1,3,nheads,1,10,10
    vid = th.ones((B,T,HD*F,H,W),device=device)
    vid0 = th.rand_like(vid)#.requires_grad_(True)
    vid1 = th.rand_like(vid)#.requires_grad_(True)

    # -- compute flow --
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    W_t = 2*wt+1
    flows = th.ones((B,1,T,W_t-1,2,nH,nW)).cuda()/2.
    flows = th.rand_like(flows)/2.+th.randint_like(flows,-3,3)+0.2
    # flows.requires_grad_(True)

    # -- exec fold fxns --
    sch = stnls.search
    search0 = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                dist_type=dist_type,
                                dilation=dil,stride0=stride0, stride1=stride1,
                                reflect_bounds=reflect_bounds,
                                full_ws=full_ws,use_adj=use_adj,
                                self_action=self_action,itype="float")
    search1 = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                dist_type=dist_type,
                                dilation=dil,stride0=stride0, stride1=stride1,
                                reflect_bounds=reflect_bounds,
                                full_ws=full_ws,use_adj=use_adj,
                                self_action=self_action,itype="float")


    # -- run search --
    vid00 = vid0.clone().requires_grad_(True)
    vid01 = vid0.clone().requires_grad_(True)
    vid10 = vid1.clone().requires_grad_(True)
    vid11 = vid1.clone().requires_grad_(True)
    flows0 = flows.clone()
    flows1 = flows.clone().requires_grad_(True)
    d0,i0 = search0(vid00,vid10,flows0)
    flows = flows.requires_grad_(True)
    d1,i1 = search1(vid01,vid11,flows1)

    # -- compute --
    dgrad = th.ones_like(d0)
    th.autograd.backward(d0,dgrad)
    th.autograd.backward(d1,dgrad)

    # -- compare --
    assert th.allclose(vid00.grad,vid01.grad,1e-3,1e-3,equal_nan=True)
    assert th.allclose(vid10.grad,vid11.grad,1e-3,1e-3,equal_nan=True)


def test_bwd_vid_gradcheck(ws,wt,ps,stride0,stride1,reflect_bounds,
                           dilation,nheads,self_action,dist_type,seed,itype):
    """

    Test the CUDA code with torch code

    Forward Pass

    """


    # -- get args --
    dil = dilation
    k,pt = -1,1
    wt = 1 if wt == 0 else wt # skip 0
    device = "cuda:0"
    full_ws = True
    set_seed(seed)

    # -- load data --
    B,HD,T,F,H,W = 1,nheads,3,1,10,10
    vid = th.ones((B,T,HD*F,H,W),device=device)
    vid0 = th.rand_like(vid).requires_grad_(True)
    vid1 = th.rand_like(vid).requires_grad_(True)

    # -- compute flow --
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    W_t = min(2*wt+1,T)
    flows = th.ones((B,HD,T,W_t-1,2,nH,nW)).cuda()/2.
    flows = th.rand_like(flows)/2.+th.randint_like(flows,-3,3)+0.2
    flows.requires_grad_(True)

    # -- exec fold fxns --
    sch = stnls.search
    # search_i = sch.NonLocalSearch(ws, wt, ps, k, nheads,
    #                             dist_type=dist_type,
    #                             dilation=dil,stride0=stride0, stride1=stride1,
    #                               reflect_bounds=True,
    #                             full_ws=full_ws,
    #                             self_action=self_action,itype=itype)
    search = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                dist_type=dist_type,
                                dilation=dil,stride0=stride0, stride1=stride1,
                                reflect_bounds=reflect_bounds,
                                full_ws=full_ws,
                                self_action=self_action,itype=itype)

    # -- tmp --
    # from stnls.testing import gradcheck
    # search_vid0 = lambda vid0: search(vid0,vid1,flows)[0]
    # search_vid0_i = lambda vid0: search_i(vid0,vid1,flows)[0]
    # num = gradcheck.get_num_jacobian(search_vid0,vid0,eps=1e-2)
    # num_i = gradcheck.get_num_jacobian(search_vid0_i,vid0,eps=1e-2)
    # ana = gradcheck.get_ana_jacobian(search_vid0,vid0,eps=1e-2)
    # ana_i = gradcheck.get_ana_jacobian(search_vid0_i,vid0,eps=1e-2)

    # print("-"*20)
    # print(num[-10:,-10:])
    # print(num_i[-10:,-10:])
    # print("-"*20)
    # print(ana[-10:,-10:])
    # print(ana_i[-10:,-10:])
    # print("-"*20)

    # -- run gradient check
    search_vid0 = lambda vid0: search(vid0,vid1,flows)[0]
    th.autograd.gradcheck(search_vid0, vid0, eps=1e-2,
                          atol=1e-2, nondet_tol=1e-7, raise_exception=True)
    search_vid1 = lambda vid1: search(vid0,vid1,flows)[0]
    th.autograd.gradcheck(search_vid1, vid1, eps=1e-3,
                          atol=1e-2, nondet_tol=1e-7, raise_exception=True)


def test_bwd_flows_gradcheck(ws,wt,ps,stride0,stride1,dilation,
                             nheads,self_action,dist_type,reflect_bounds,seed):
    """

    Test the CUDA code with torch code

    Forward Pass

    """


    # -- get args --
    dil = dilation
    k,pt = -1,1
    if wt == 0: return # skip wt == 0
    device = "cuda:0"
    full_ws = True
    set_seed(seed)

    # -- load video --
    B,HD,T,F,H,W = 1,nheads,5,1,10,10
    vid = th.ones((B,T,HD*F,H,W),device=device)
    vid0,vid1 = vid.clone(),vid.flip(-1).clone()
    vid0 = th.rand_like(vid)-1.5
    vid1 = th.rand_like(vid)#-0.5
    B,T,F,H,W = vid0.shape

    # -- load flows --
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    W_t = min(2*wt+1,T)
    flows = th.ones((B,HD,T,W_t-1,2,nH,nW)).cuda()/2.
    flows = th.rand_like(flows)/5.+th.randint_like(flows,-2,2)+0.5
    not_int = th.all(th.abs(flows.round() - flows)>1e-5).item()
    assert not_int,"Gradcheck only works _not_ near an int."
    # unstable at any "ints" including the search; if stride1 is not int,
    # then an false alarm is raised

    # away from int
    flows.requires_grad_(True)

    # -- exec fold fxns --
    sch = stnls.search
    search = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                   dist_type=dist_type,
                                   dilation=dil,stride0=stride0, stride1=stride1,
                                   reflect_bounds=reflect_bounds,
                                   full_ws=full_ws,
                                   self_action=self_action,itype="float")

    # -- immersive --
    # from stnls.testing import gradcheck
    # search_flows = lambda flows: search(vid0,vid1,flows)[0]
    # num = gradcheck.get_num_jacobian(search_flows,flows,eps=1e-3)
    # ana = gradcheck.get_ana_jacobian(search_flows,flows,eps=1e-4)
    # args = th.where(th.abs(num - ana)>1e-2)
    # print(num[:10,:10])
    # print(ana[:10,:10])
    # print(args)
    # print(num[args][:10])
    # print(ana[args][:10])

    # print(num[:2,100:115])
    # print(ana[:2,100:115])

    # print(num[764:767,35:45])
    # print(ana[764:767,35:45])

    # print(num[-10:,-10:])
    # print(ana[-10:,-10:])
    # print(th.where(th.abs(num-ana)>1e-3))


    # -- gradient check --
    search_flows = lambda flows: search(vid0,vid1,flows)[0]
    th.autograd.gradcheck(search_flows, flows, eps=1e-2,
                          atol=1e-2, nondet_tol=1e-7, raise_exception=True)

    search_flows = lambda flows: search(vid0,vid1,flows)[1]
    th.autograd.gradcheck(search_flows, flows, eps=1e-2,
                          atol=1e-2, nondet_tol=1e-7, raise_exception=True)


def test_fwd_topk(ws,wt,ps,stride0,stride1,dilation,reflect_bounds,
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
    full_ws = True
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    topk_mode = "each"
    itype = "float"
    nheads = 1

    # -- load data --
    vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid.to(device)[:,:5,:3,:,:].contiguous()
    vid = repeat(vid,'b t c h w -> b t (r c) h w',r=12)[:,:32].contiguous()
    vid /= vid.max()
    vid0 = th.rand_like(vid)-0.5
    vid1 = th.rand_like(vid)-0.5

    # -- compute flow --
    B,T,F,H,W = vid.shape
    W_t = 2*wt+1
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    flows = 2*th.rand((B,T,W_t-1,2,nH,nW)).to(vid0.device)

    # -- exec fold fxns --
    k = ws*ws*W_t
    search = stnls.search.NonLocalSearch(ws, wt, ps, k, nheads,
                                         dilation=dil,stride0=stride0, stride1=stride1,
                                         reflect_bounds=reflect_bounds,full_ws=full_ws,
                                         self_action=None,
                                         dist_type=dist_type,topk_mode="all",
                                         itype=itype)

    # -- exec --
    dists,inds = search(vid0,vid1,flows)
    delta = dists[...,1:] - dists[...,:-1]
    if dist_type == "l2":
        assert th.all(delta>=0).item()
    else:
        assert th.all(delta<=0).item()

def test_fwd_anchor(ws,wt,ps,stride0,stride1,dilation,
                    self_action,dist_type,reflect_bounds,seed):

    """

    Test the CUDA code with torch code

    Forward Pass

    """

    # -- init vars --
    device = "cuda:0"
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    topk_mode = "each"
    itype = "float"
    pt,nheads = 1,1
    full_ws = True
    set_seed(seed)

    # -- load data --
    vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid.to(device)[:,:5,:3,::2,::2].contiguous()
    vid = repeat(vid,'b t c h w -> b t (r c) h w',r=12)[:,:,:32].contiguous()
    vid /= vid.max()
    vid0 = th.rand_like(vid)-0.5
    vid1 = th.rand_like(vid)-0.5

    # -- compute flow --
    B,T,F,H,W = vid.shape
    W_t = 2*wt+1
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    flows = 2*th.rand((B,T,W_t-1,2,nH,nW)).to(vid0.device)
    # flows = th.zeros_like(flows)

    # -- exec fold fxns --
    k0 = ws*ws
    search0 = stnls.search.NonLocalSearch(ws, wt, ps, -1, nheads,
                                          dilation=dilation,
                                          stride0=stride0, stride1=stride1,
                                          reflect_bounds=reflect_bounds,full_ws=False,
                                          self_action=None,
                                          dist_type=dist_type,topk_mode=topk_mode,
                                          itype=itype)
    k1 = 3
    search1 = stnls.search.NonLocalSearch(ws, wt, ps, k1, nheads,
                                          dilation=dilation,
                                          stride0=stride0, stride1=stride1,
                                          reflect_bounds=reflect_bounds,full_ws=False,
                                          self_action="anchor_each",
                                          dist_type=dist_type,topk_mode=topk_mode,
                                          itype=itype)
    k2 = 5
    search2 = stnls.search.NonLocalSearch(ws, wt, ps, k2, nheads,
                                          dilation=dilation,
                                          stride0=stride0, stride1=stride1,
                                          reflect_bounds=reflect_bounds,full_ws=True,
                                          self_action="anchor_each",
                                          dist_type=dist_type,topk_mode=topk_mode,
                                          itype=itype)
    k3 = 8
    search3 = stnls.search.NonLocalSearch(ws, wt, ps, k3, nheads,
                                          dilation=dilation,
                                          stride0=stride0, stride1=stride1,
                                          reflect_bounds=reflect_bounds,full_ws=True,
                                          self_action="anchor",
                                          dist_type=dist_type,topk_mode=topk_mode,
                                          itype=itype)
    k4 = 8
    search4 = stnls.search.NonLocalSearch(ws, wt, ps, k4, nheads,
                                          dilation=dilation,
                                          stride0=stride0, stride1=stride1,
                                          reflect_bounds=reflect_bounds,full_ws=True,
                                          self_action="anchor_and_remove_ref_frame",
                                          dist_type=dist_type,topk_mode="each",
                                          itype=itype)



    # -- exec --
    HD = nheads
    vshape = (B,HD,T,nH,nW,W_t)

    dists0,inds0 = search0(vid0,vid1,flows)
    dists0,inds0 = dists0.view(vshape+(k0,)),inds0.view(vshape+(k0,3,))
    dists0 = dists0[...,:,ws//2*ws+ws//2]
    inds0= inds0[...,:,ws//2*ws+ws//2,:]

    dists1,inds1 = search1(vid0,vid1,flows)
    dists1,inds1 = dists1.view(vshape+(k1,)),inds1.view(vshape+(k1,3,))
    dists1 = dists1[...,:,0]
    inds1= inds1[...,:,0,:]

    dists2,inds2 = search2(vid0,vid1,flows)
    dists2,inds2 = dists2.view(vshape+(k2,)),inds2.view(vshape+(k2,3,))
    dists2 = dists2[...,:,0]
    inds2= inds2[...,:,0,:]

    dists3,inds3 = search3(vid0,vid1,flows)
    dists3,inds3 = dists3.view(vshape+(k3,)),inds3.view(vshape+(k3,3,))
    dists3 = dists3[...,:,0]
    inds3= inds3[...,:,0,:]

    vshape = (B,HD,T,nH,nW,W_t-1)
    dists4,inds4 = search4(vid0,vid1,flows)
    dists4,inds4 = dists4.view(vshape+(k4,)),inds4.view(vshape+(k4,3,))
    dists4 = dists4[...,:,0]
    inds4 = inds4[...,:,0,:]

    # print(th.stack([dists0[...,0],dists1[...,0],dists2[...,0],dists3[...,0]],-1))
    # print(th.stack([inds0[...,0,:],inds1[...,0,:],inds2[...,0,:],inds3[...,0,:]],-2))

    # -- check all pairwise --
    dists = [dists0,dists1,dists2]
    inds = [inds0,inds1,inds2]
    for i in range(3):
        for j in range(3):
            if i == j: continue
            assert th.allclose(dists[i],dists[j],1e-3,1e-3,equal_nan=True)
            assert th.allclose(inds[i],inds[j],1e-3,1e-3,equal_nan=True)

    # -- check all against "anchor" --
    for i in range(3):
        assert th.allclose(dists[i][...,0],dists3[...,0],1e-3,1e-3,equal_nan=True)
        assert th.allclose(inds[i][...,0,:],inds3[...,0,:],1e-3,1e-3,equal_nan=True)

    # -- check against flow --
    def reflect_bounds(flow,i,L):
        args0 = th.where(flow[:,i] > (L-1))
        args1 = th.where(flow[:,i] < 0)
        flow[:,i][args0] = 2*(L-1) - flow[:,i][args0]
        flow[:,i][args1] = -flow[:,i][args1]
    grid = stnls.nn.index_grid(nH,nW).flip(1)*stride0
    for i in range(3):
        inds_i = inds[i]
        for ti in range(T):
            for si in range(W_t):
                ind = inds_i[:,0,ti,:,:,si,1:]
                if si > 0:
                    flow = flows[:,ti,si-1].flip(1).clone() + grid
                else:
                    flow = th.zeros_like(flows[:,ti,0]).flip(1) + grid

                # -- reflect --
                reflect_bounds(flow,0,H)
                reflect_bounds(flow,1,W)

                # -- normalize --
                flow = flow - grid

                # -- shaping --
                flow = rearrange(flow,'b i h w -> b h w i')
                assert th.allclose(ind,flow,1e-3,1e-3,equal_nan=True)

    inds_i = inds4
    for ti in range(T):
        for si in range(W_t-1):
            ind = inds_i[:,0,ti,:,:,si,1:]
            flow = flows[:,ti,si].flip(1).clone() + grid

            # -- reflect --
            reflect_bounds(flow,0,H)
            reflect_bounds(flow,1,W)

            # -- normalize --
            flow = flow - grid

            # -- shaping --
            flow = rearrange(flow,'b i h w -> b h w i')
            print(th.cat([ind,flow],-1))

            diff = th.mean(th.abs(ind - flow)).item()
            assert th.allclose(ind,flow,1e-3,1e-3,equal_nan=True)

