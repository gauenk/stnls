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

# -- test func --
from torch.nn.functional import fold,unfold,pad
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/tile/non_local_stack")

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_data(dnames,ext,device="cuda:0"):
    vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid.to(device)[:,:5,].contiguous()
    vid = repeat(vid,'b t c h w -> b t (r c) h w',r=12)[:,:32].contiguous()
    # vid = repeat(vid[:,0],'b c h w -> b t c h w',t=5)
    vid = vid[:1,:,:1].contiguous()
    vid /= vid.max()
    return vid

def pytest_generate_tests(metafunc):
    # only for stride1 == 1
    test_lists = {"wt":[2],"ws":[15],"k":[5],"ps":[7],
                  "stride0":[4],"stride1":[1],"dilation":[1],
                  "nheads":[1],"anchor_self":[True],
                  "full_ws":[True],"dist_type":["l2"],"seed":[0]}
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
    reflect_bounds = True
    set_seed(seed)
    itype_fwd = "float"

    # -- load data --
    vid = get_data(dnames,ext)

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(run_flow,clean_flow,vid,vid,0.)
    flows.fflow = th.clamp(5*th.randn_like(flows.fflow),-5,5).round()
    flows.bflow = th.clamp(5*th.randn_like(flows.bflow),-5,5).round()
    # flows.fflow = th.clamp(5*th.zeros_like(flows.fflow),-5,5)
    # flows.bflow = th.clamp(5*th.zeros_like(flows.bflow),-5,5)

    # -- exec fold fxns --
    sch = stnls.search
    search = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                dist_type=dist_type, dilation=dil,
                                stride0=stride0, stride1=stride1,
                                reflect_bounds=reflect_bounds, full_ws=full_ws,
                                anchor_self=anchor_self,itype_fwd=itype_fwd)
    dists,inds = search(vid,vid,flows.fflow,flows.bflow)
    weights = th.exp(-dists/10)
    # weights[...] = 1.
    stacking = stnls.tile.NonLocalStack(ps=ps,stride0=stride0,
                                        reflect_bounds=reflect_bounds,
                                        itype_fwd=itype_fwd)
    stacking_gt = stnls.tile.NonLocalStackGt(ps=ps,stride0=stride0,
                                             reflect_bounds=reflect_bounds)

    # -- [testing] search --
    stack_te = stacking(vid,weights,inds)
    stack_gt = stacking_gt(vid,weights,inds)
    # print("te: ",th.any(th.isnan(stack_te)).item())
    # print("gt: ",th.any(th.isnan(stack_gt)).item())
    # print("te > 0: ",th.any(stack_te > 0).item())
    # # print(th.stack([stack_te,stack_gt],-1))
    # print(stack_te[0,0,1,0])
    # print(stack_gt[0,0,1,0])

    # -- [testing] viz --
    # diff = th.mean((stack_te - stack_gt)**2,dim=(1,2))
    # diff /= diff.max()
    # stnls.utils.vid_io.save_video(diff,"./output/tests/tile/","stack_diff")
    # K = stack_gt.shape[2]
    # for ki in range(K):
    #     stnls.utils.vid_io.save_video(stack_gt[:,0,ki,:1],
    #                                   "./output/tests/tile/","stack_gt_%d"%ki)
    #     stnls.utils.vid_io.save_video(stack_te[:,0,ki,:1],
    #                                   "./output/tests/tile/","stack_te_%d"%ki)

    # -- [self similar error] --
    # K = stack_gt.shape[2]
    # diffs = np.zeros((K,K))
    # for ki in range(K):
    #     for kj in range(K):
    #         mse = th.mean((stack_gt[:,:,ki] - stack_gt[:,:,kj])**2).item()
    #         psnr = -10*np.log10(mse)
    #         diffs[ki,kj] = psnr
    # print(diffs)

    # -- pick tolerance --
    mean_tol = 1e-3
    max_tol = 1e-5

    # -- compare --
    isinf = th.isinf(stack_gt)
    issmall = stack_gt < 1e-4
    args0 = th.where(th.logical_not(th.logical_or(isinf,issmall))) # remove invalid
    diff = th.abs(stack_te - stack_gt) / (stack_gt.abs()+1e-8)

    dargs = th.where(diff>0.5)
    # print(dargs)
    # print(stack_gt[dargs])
    # print(stack_te[dargs])

    # -- test --
    error = diff[args0].mean().item()
    # print(error)
    if error > mean_tol: print("error: ",error)
    assert error < mean_tol

    max_error = diff[args0].max().item()
    # print(max_error)
    if max_error > max_tol: print("max error: ",max_error)
    assert max_error < max_tol


def test_bwd(ws,wt,k,ps,stride0,stride1,dilation,
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
    reflect_bounds = True
    itype_fwd = "float"
    itype_bwd = "float"
    set_seed(seed)

    # -- load data --
    vid = get_data(dnames,ext)
    # vid[...] = 1


    # -- compute flow --
    flows = stnls.flow.get_flow_batch(run_flow,clean_flow,vid,vid,0.)
    # flows.fflow = th.clamp(5*th.randn_like(flows.fflow),-5,5)
    # flows.bflow = th.clamp(5*th.randn_like(flows.bflow),-5,5)
    flows.fflow = th.clamp(5*th.zeros_like(flows.fflow),-5,5).round()
    flows.bflow = th.clamp(5*th.zeros_like(flows.bflow),-5,5).round()

    # -- exec fold fxns --
    sch = stnls.search
    search = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                dist_type=dist_type, dilation=dil,
                                stride0=stride0, stride1=stride1,
                                reflect_bounds=True,#reflect_bounds,
                                full_ws=full_ws,anchor_self=anchor_self,
                                itype_fwd=itype_fwd)
    dists,inds = search(vid,vid,flows.fflow,flows.bflow)
    # dists[...] = 1

    # -- create grad pairs --
    dists_te = dists.clone().requires_grad_(True)
    dists_gt = dists.clone().requires_grad_(True)
    vid_te = vid.clone().requires_grad_(True)
    vid_gt = vid.clone().requires_grad_(True)
    weights_te = dists_te#th.exp(-dists_te/10)
    weights_gt = dists_gt#th.exp(-dists_gt/10)

    # -- init stacking --
    # stacking = stnls.tile.NonLocalStack(ps=ps,stride0=stride0,
    #                                     reflect_bounds=reflect_bounds)
    stacking = stnls.tile.NonLocalStack(ps=ps,stride0=stride0,
                                        reflect_bounds=reflect_bounds,
                                        itype_fwd=itype_fwd,itype_bwd=itype_bwd)
    stacking_gt = stnls.tile.NonLocalStackGt(ps=ps,stride0=stride0,
                                             reflect_bounds=reflect_bounds)

    # -- [testing] search --
    stack_te = stacking(vid_te,weights_te,inds)
    stack_gt = stacking_gt(vid_gt,weights_gt,inds)
    # print("te: ",th.any(th.isnan(stack_te)).item())
    # print("gt: ",th.any(th.isnan(stack_gt)).item())
    # print("te > 0: ",th.any(stack_te > 0).item())
    # print(th.stack([stack_te,stack_gt],-1))

    # print(stack_te[0,0,1,0,0][30:34,30:34])
    # print(stack_gt[0,0,1,0,0][30:34,30:34])

    # -- autograd --
    stack_grad = th.randn_like(stack_te)
    # stack_grad = th.ones_like(stack_te)
    th.autograd.backward(stack_te,stack_grad)
    th.autograd.backward(stack_gt,stack_grad)

    # -- [testing] viz --
    # print("dstack: ")
    # dstack = th.stack([dists_te.grad,dists_gt.grad,dists_gt.grad/dists_te.grad],-1)
    # # print(dstack.shape)
    # print(dstack[0,0,32+10:32+15])
    # print("vstack: ")
    # vstack = th.stack([vid_te.grad,vid_gt.grad,vid_gt.grad/vid_te.grad],-1)
    # # print(vstack.shape)
    # # print(vstack[0,0,0,:10,:10])
    # print(vstack[0,0,0,30:34,30:34])
    # # print(vid_te.grad)
    # # print(vid_gt.grad)
    # # print(vid_gt.grad/vid_te.grad)
    # diff = th.mean((vid_te.grad - vid_gt.grad)**2,dim=(0,2),keepdim=True)
    # # print("v diff max: ",diff.max())
    # diff /= diff.max()
    # stnls.utils.vid_io.save_video(diff,"./output/tests/tile/","d_vid_diff")
    # stnls.utils.vid_io.save_video(vid_te.grad,"./output/tests/tile/","d_vid_te")
    # stnls.utils.vid_io.save_video(vid_gt.grad,"./output/tests/tile/","d_vid_gt")

    # -- check each grad --
    d_dists = th.mean((dists_te.grad - dists_gt.grad)**2/dists_te.grad.abs().max())
    d_vid = th.mean((vid_te.grad - vid_gt.grad)**2)
    assert d_dists < 1e-10
    assert d_vid < 1e-10
