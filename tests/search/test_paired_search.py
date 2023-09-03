
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
from torchvision.utils import make_grid,save_image

# -- paths --
SAVE_DIR = Path("./output/tests/paired_search")

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

def run_compare(tensor_gt,tensor_te,mean_tol,max_tol,small_tol=1e-3):

    # -- compute diffs --
    cond_a = th.logical_not(th.isinf(tensor_gt))
    cond_b = tensor_gt.abs() > small_tol
    args0 = th.where(th.logical_and(cond_a,cond_b)) # remove all inf
    diff = th.abs(tensor_te - tensor_gt) / (tensor_gt.abs()+1e-4)
    diff = diff[args0]

    # -- viz --
    args1 = th.where(diff.abs() > 1e-3)
    if len(tensor_gt[args0][args1]) < 20: # allow a few to be different
        diff = diff[th.where(diff.abs() < 1e-3)]
    print(len(tensor_gt[args0][args1]))
    print(tensor_gt[args0][args1])
    print(tensor_te[args0][args1])
    if len(tensor_gt[args0][args1]) > 0:
        print(tensor_gt[args0][args1][0].item())
        print(tensor_te[args0][args1][0].item())


    # -- test --
    error = diff.mean().item()
    if error > mean_tol: print("error: ",error)
    assert error < mean_tol

    max_error = diff.max().item()
    if max_error > max_tol: print("max error: ",max_error)
    assert max_error < max_tol


def pytest_generate_tests(metafunc):
    test_lists = {"ps":[1],"stride0":[1],"stride1":[1],
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

    def make_igrid(x):
        b,t,_,h,w = x.shape
        grid_y, grid_x = th.meshgrid(th.arange(0, h, dtype=x.dtype, device=x.device),
                                     th.arange(0, w, dtype=x.dtype, device=x.device))
        grid = th.stack((grid_x, grid_y), 0).float()[None,:]  # 1, 2, W(x), H(y)
        grid.requires_grad = False
        return grid

    # -- load data --
    vid = get_data(dnames,ext)
    vid = th.cat([vid,vid],1)
    # vid0 = vid#+th.randn_like(vid)
    vid = vid[...,:,:,:].contiguous()
    B,T,C,H,W = vid.shape
    # vid = make_igrid(vid).repeat(3,1,1,1).view(1,3,2,H,W)
    # vid = vid[...,:,:,:] / (H-1)
    # vid[...,-1,:,:] = 0
    # vid = th.ones_like(vid)
    # vid = th.randn_like(vid)
    vid0 = vid
    # vid0 = th.ones_like(vid)
    # vid1 = vid.clone()
    # vid0 = vid.clone()
    # vid0 = th.randn_like(vid).round(decimals=3)
    vid0 = th.randn_like(vid)
    vid1 = th.randn_like(vid)
    # vid1 = th.ones_like(vid)

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)
    M = 10
    # flows.fflow = th.clamp(th.ones_like(flows.fflow),-M,M).round()
    # flows.bflow = th.clamp(th.ones_like(flows.bflow),-M,M).round()
    # flows.fflow = th.clamp(M*th.randn_like(flows.fflow),-M,M).round()/M*2
    # flows.bflow = th.clamp(M*th.randn_like(flows.bflow),-M,M).round()/M*2
    flows.fflow = th.clamp(th.randn_like(flows.fflow),-M,M)
    flows.bflow = th.clamp(th.randn_like(flows.bflow),-M,M)
    # flows.fflow[:,2] = flows.fflow[:,1]
    # flows.bflow[:,1] = flows.fflow[:,1]
    # flows.bflow[:,2] = flows.bflow[:,1]
    # flows.fflow[:,1] = 0
    # flows.bflow[:,1] = th.ones_like(flows.bflow[:,1])
    # flows.bflow[:,2] = -th.ones_like(flows.bflow[:,1])
    # flows.bflow = th.round(flows.bflow,decimals=2)
    # flows.bflow = th.round(flows.bflow,decimals=2)

    # flows.fflow = th.zeros_like(flows.fflow)
    # flows.bflow = th.zeros_like(flows.bflow)
    # flows.fflow = th.round(flows.fflow,decimals=2)
    # flows.bflow = th.round(flows.bflow,decimals=2)
    flows.fflow[:,-1,...] = 0
    flows.bflow[:,0,...] = 0

    # -- unpack image --
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
    dists_gt,inds_gt = search_gt(vid0,vid1,flows.fflow,flows.bflow)
    # dists_gt = th.round(dists_gt,decimals=2)
    # inds_gt = th.round(inds_gt,decimals=3)
    th.cuda.synchronize()

    # -- [testing] search --
    dtype = th.float if itype_fwd == "float" else th.int
    acc_flows = stnls.nn.accumulate_flow(flows,dtype=dtype)
    # acc_flows.fflow = th.round(acc_flows.fflow,decimals=4)
    # acc_flows.bflow = th.round(acc_flows.bflow,decimals=4)
    dists_te,inds_te = search_te.paired_vids(vid0, vid1, acc_flows, wt)
    # dists_te = th.round(dists_te,decimals=2)
    # inds_te = th.round(inds_te,decimals=3)

    for t in range(2):
        print("-"*10 + ("t: %d" % t)  + "-"*10)
        txt = ["W","H"]
        for i in range(2):
            print(txt[i])
            print(flows.fflow[0,t,i,:3,:3])

    for t in range(2):
        print("-"*10 + ("t: %d" % t)  + "-"*10)
        txt = ["W","H"]
        for i in range(2):
            print(txt[i])
            print(acc_flows.fflow[0,0,t,i,:3,:3])
        # print(flows.fflow[0,t,1,:2,:2])
        # print(acc_flows.fflow[0,0,t,1,:2,:2])

    # -- viz --
    K = 27
    print(inds_te.shape)
    diff = th.mean(1.*(inds_te[0,0] - inds_gt[0,0])**2,dim=(-1,))
    args = th.where(diff>1e-13)
    diff = rearrange(diff,'(t h w) k -> k t h w',h=16,w=16)
    qi = -256+64

    dinds = []
    for i in range(3):
        cat_inds = th.stack([inds_gt[0,0,...,i][args],inds_te[0,0,...,i][args]],-1)
        dinds.append(cat_inds)
    dinds = th.cat(dinds,-1)
    print(dinds)
    print(dinds[:10])
    # for j in range(4):
    #     print(dinds[0,j+2].item())

    # diff_grid = make_grid(diff,nrow=K,pad_value=1.)
    # diff_grid /= diff_grid.max()
    # print(diff_grid.shape)
    # save_image(diff_grid,SAVE_DIR/"inds_diffs.png")
    # stnls.utils.vid_io.save_video(diff_grid,SAVE_DIR,"inds_diffs")

    #
    #
    # -- info --
    #
    #

    # -- compute diffs --
    tensor_gt = dists_gt
    tensor_te = dists_te
    args0 = th.where(th.logical_not(th.isinf(tensor_gt))) # remove all inf
    diff = th.abs(tensor_te - tensor_gt) / (tensor_gt.abs()+1e-5)
    # args1 = th.where(tensor_gt[args0].abs() > 1e-4)
    diff = diff[args0]
    args1 = th.where(diff.abs() > 1e-3)

    print("dists: ")
    print(tensor_gt[args0][args1])
    print(tensor_te[args0][args1])

    tensor_gt = inds_gt
    tensor_te = inds_te
    for i in range(3):
        print(tensor_gt[...,i][args0][args1])
        print(tensor_te[...,i][args0][args1])
    # print(vid0[0,0,:,0,60].item())

    # -- compare --
    mean_tol = 5e-3
    max_tol = 1e-3
    sm_tol = 1e-2
    run_compare(inds_gt,inds_te,mean_tol,max_tol,sm_tol)
    max_tol = 1e-2
    run_compare(dists_gt,dists_te,mean_tol,max_tol,sm_tol)


def test_bwd(ws,wt,k,ps,stride0,stride1,dilation,k_agg,
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
    vid = th.cat([vid,vid],1) # 6 frames
    # vid = th.ones_like(vid)

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)
    M = 1
    # flows.fflow = th.clamp(th.randn_like(flows.fflow),-M,M).round()/2.
    # flows.bflow = th.clamp(th.randn_like(flows.bflow),-M,M).round()/2.
    # flows.fflow = th.clamp(th.randn_like(flows.fflow),-M,M).round()/2.
    # flows.bflow = th.clamp(th.randn_like(flows.bflow),-M,M).round()/2.
    flows.fflow = th.clamp(M*th.ones_like(flows.fflow),-M,M).round()/5.
    flows.bflow = th.clamp(M*th.ones_like(flows.bflow),-M,M).round()/5.
    # flows.fflow = th.zeros_like(flows.fflow)
    # flows.bflow = th.zeros_like(flows.bflow)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape
    itype_fwd = "float"
    itype_bwd = "float"

    # -- init data --
    vid0 = th.randn_like(vid)
    vid1 = th.randn_like(vid)
    # vid0 = th.ones_like(vid)#vid+th.randn_like(vid)
    # vid1 = th.ones_like(vid)#vid.clone()
    vid0_te = vid0.clone().requires_grad_(True)
    vid0_gt = vid0.clone().requires_grad_(True)
    vid1_te = vid1.clone().requires_grad_(True)
    vid1_gt = vid1.clone().requires_grad_(True)
    fflow_gt = flows.fflow.clone().requires_grad_(True)
    bflow_gt = flows.bflow.clone().requires_grad_(True)
    fflow_te = flows.fflow.clone().requires_grad_(True)
    bflow_te = flows.bflow.clone().requires_grad_(True)

    # -- exec fold fxns --
    sch = stnls.search
    search_te = sch.PairedSearch(ws, ps, k, nheads, dist_type=dist_type,
                                 dilation=dil,stride0=stride0, stride1=stride1,
                                 reflect_bounds=reflect_bounds,
                                 full_ws=True,full_ws_time=True,
                                 anchor_self=anchor_self,use_adj=use_adj,
                                 itype_fwd=itype_fwd,itype_bwd=itype_bwd,
                                 normalize_bwd=False)
    search_gt = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                   dist_type=dist_type,
                                   dilation=dil,stride0=stride0, stride1=stride1,
                                   reflect_bounds=reflect_bounds,
                                   full_ws=True,full_ws_time=True,
                                   anchor_self=anchor_self,use_adj=use_adj,
                                   itype_fwd=itype_fwd,itype_bwd=itype_bwd,
                                   normalize_bwd=False)

    # -- [groundtruth] search --
    dists_gt,inds_gt = search_gt(vid0_gt,vid1_gt,fflow_gt,bflow_gt)
    th.cuda.synchronize()
    # dists_gt = dists_gt[:,:,256:512]
    # inds_gt = inds_gt[:,:,256:512]

    # -- [testing] search --
    dtype = th.float
    acc_flows = stnls.nn.accumulate_flow(fflow_te,bflow_te,dtype=dtype)
    dists_te,inds_te = search_te.paired_vids(vid0_te, vid1_te, acc_flows, wt)
    # dists_te = dists_te[:,:,256:512]
    # inds_te = inds_te[:,:,256:512]


    # -- compare --
    mean_tol = 5e-3
    max_tol = 1e-2
    sm_tol = 1e-2
    run_compare(inds_gt,inds_te,mean_tol,max_tol,sm_tol)
    run_compare(dists_gt,dists_te,mean_tol,max_tol,sm_tol)

    # -- backprop inds --
    inds_grad = th.randn_like(inds_gt)
    th.autograd.backward(inds_gt,inds_grad,retain_graph=True)
    th.autograd.backward(inds_te,inds_grad,retain_graph=True)

    # -- viz --
    print(th.any(th.isnan(bflow_gt.grad)))
    print(th.any(th.isnan(bflow_te.grad)))
    print(th.stack([fflow_te.grad[0],fflow_gt.grad[0]],-1))

    # -- flow grads --
    mean_tol = 5e-3
    max_tol = 1e-2
    sm_tol = 1e-2
    _grads_gt = [fflow_gt.grad,bflow_gt.grad]
    _grads_te = [fflow_te.grad,bflow_te.grad]
    for idx,(grads_gt,grads_te) in enumerate(zip(_grads_gt,_grads_te)):
        # print(th.any(th.isnan(grads_gt)),th.any(th.isnan(grads_te)))
        run_compare(grads_gt,grads_te,mean_tol,max_tol,sm_tol)

    # -- backprop vids --
    dists_grad = th.randn_like(dists_gt)
    th.autograd.backward(dists_gt,dists_grad,retain_graph=True)
    th.autograd.backward(dists_te,dists_grad,retain_graph=True)

    # -- vid grads --
    mean_tol = 5e-3
    max_tol = 1e-2
    _grads_gt = [vid0_gt.grad,vid1_gt.grad]
    _grads_te = [vid0_te.grad,vid1_te.grad]
    for idx,(grads_gt,grads_te) in enumerate(zip(_grads_gt,_grads_te)):
        run_compare(grads_gt,grads_te,mean_tol,max_tol)


