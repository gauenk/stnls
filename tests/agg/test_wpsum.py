"""

Weighted Patch Sum

Verbose Psuedo-Code:

   patches_i = unfold_k(b2,nlInds_cu).type(th.float64)
   patches_i = rearrange(patches_i,'n k 1 c h w -> n k (c h w)')
   zpatches = []
   for ki in range(k):
      yi = softmax(dists[ki])
      zi = th.sum(yi * patches_i,1).type(th.float32) # n (c h w), this code!
      zpatches.append(zi)
   zpatches = th.stack(zpatches)

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
from torchvision import utils as tv_utils

# -- stnls --
import stnls

# -- paths --
SAVE_DIR = Path("./output/tests/wpsum")

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
    test_lists = {"ps":[5],"stride0":[1,2,3],"pt":[1],
                  "dilation":[2],"K":[5,10],"nheads":[1,2],
                  "reflect_bounds":[True,False],
                  "seed":[0,1,2],"itype":["float","int"]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def test_fwd(K,ps,pt,stride0,dilation,
             reflect_bounds,nheads,itype,seed):

    """

    Test the CUDA code with torch code

    Forward Pass

    """

    # -- init --
    device = "cuda:0"
    set_seed(seed)

    # -- load data --
    B,HD,T,F,H,W = 1,nheads,4,3,16,16
    vid = th.rand((B,T,F*HD,H,W),device=device).float()
    # vid = th.ones_like(vid)

    # -- load weights --
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    weights = th.rand((B,HD,T,nH,nW,K)).to(device)
    # weights = th.ones_like(weights)
    # vid[0,0,:,0,0] = 0

    # -- load flows --
    flows = th.ones((B,HD,T,nH,nW,K,3))+0.1
    flows = th.rand_like(flows)/2.+1.1
    tgrid = th.arange(0,T).view(1,1,T,1,1,1)
    flows[...,0] = th.randint(0,T,size=flows[...,0].shape)-tgrid
    flows[...,1:] = th.rand_like(flows[...,1:])/2.+0.2
    not_int = th.all(th.abs(flows[...,1:].round() - flows[...,1:])>1e-5).item()
    # flows[...,1:] = flows[...,1:].int()
    # flows[...,0] = flows[...,0].round().int()
    if itype == "float":
        assert not_int,"Gradcheck only works _not_ near an int."
    else:
        flows = flows.round().int()
    flows = flows.to(vid.device)

    # -- exec fold fxns --
    agg_gt = stnls.agg.NonLocalStack(ps=ps,stride0=stride0,pt=pt,
                                     reflect_bounds=reflect_bounds,
                                     itype=itype)
    out_gt = th.sum(agg_gt(vid,weights,flows),2)
    # print("out_gt.shape: ",out_gt.shape)

    # -- exec fold fxns --
    # agg_gt = stnls.agg.NonLocalStack(ps=ps,stride0=stride0,pt=pt,
    #                                  reflect_bounds=reflect_bounds,
    #                                  itype="int")
    # out_te = th.sum(agg_gt(vid,weights,flows),2)
    agg_te = stnls.agg.WeightedPatchSum(ps=ps,stride0=stride0,pt=pt,
                                        reflect_bounds=reflect_bounds,
                                        itype=itype)
    out_te = agg_te(vid,weights,flows)
    # print("out_te.shape: ",out_te.shape)

    # print(th.where(th.abs(out_gt-out_te)>1e-3))
    # print(out_gt[0,0,0,0,-5:,-5:])
    # print(out_te[0,0,0,0,-5:,-5:])
    # print((out_gt[0,0,0,0] - out_te[0,0,0,0]).abs()>1e-3)

    assert th.allclose(out_te,out_gt,1e-3,1e-3,equal_nan=True)

def test_bwd(K,ps,pt,stride0,dilation,
             reflect_bounds,nheads,itype,seed):

    """

    Test the CUDA code with torch code

    Forward Pass

    """

    # -- get args --
    device = "cuda:0"
    set_seed(seed)

    # -- load data --
    B,HD,T,F,H,W = 1,nheads,4,3,8,8
    vid = th.rand((B,HD,T,F,H,W),device=device).float()
    # vid = th.ones_like(vid)
    vid = vid.requires_grad_(True)

    # -- load weights --
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    weights = th.rand((B,HD,T,nH,nW,K)).to(device)
    # weights = th.ones_like(weights)
    # # weights[...,1:] = 0
    weights = weights.requires_grad_(True)

    # -- load flows --
    flows = th.ones((B,HD,T,nH,nW,K,3))+0.1
    flows = th.rand_like(flows)/2.+1.1
    tgrid = th.arange(0,T).view(1,1,T,1,1,1)
    flows[...,0] = th.randint(0,T,size=flows[...,0].shape)-tgrid
    # flows[...,1:]=th.rand_like(flows[...,1:])/2.+0.2+th.randint_like(flows[...,1:],-3,3)
    flows[...,1:] = th.rand_like(flows[...,1:])/2.+0.2
    not_int = th.all(th.abs(flows[...,1:].round() - flows[...,1:])>1e-5).item()
    # flows[...,0] = th.zeros_like(flows[...,0])
    # flows[...,1:] = th.zeros_like(flows[...,1:])
    flows = flows.to(vid.device)
    flows_t,flows_sp = flows[...,[0]],flows[...,1:]
    if itype == "float":
        assert not_int,"Gradcheck only works _not_ near an int."
    else:
        flows = flows.round().int()

    # -- exec fold fxns --
    agg = stnls.agg.WeightedPatchSum(ps=ps,stride0=stride0,
                                     reflect_bounds=reflect_bounds,
                                     itype=itype)

    # -- gradcheck --
    stack_vid = lambda vid: agg(vid,weights,flows)
    th.autograd.gradcheck(stack_vid, vid, eps=1e-2,
                          atol=1e-2, nondet_tol=1e-5, raise_exception=True)

    stack_weights = lambda weights: agg(vid,weights,flows)
    th.autograd.gradcheck(stack_weights, weights, eps=1e-2,
                          atol=1e-2, nondet_tol=1e-5, raise_exception=True)

    if itype == "float":
        flows_sp = flows_sp.requires_grad_(True)
        def stack_flows(flows_sp):
            flows = th.cat([flows_t,flows_sp],-1)
            return agg(vid,weights,flows)
        th.autograd.gradcheck(stack_flows, flows_sp, eps=1e-2,
                              atol=1e-2, nondet_tol=1e-7, raise_exception=True)
        # from stnls.testing import gradcheck
        # ana = gradcheck.get_ana_jacobian(stack_flows,flows_sp,eps=1e-2)
        # num = gradcheck.get_num_jacobian(stack_flows,flows_sp,eps=1e-2,nreps=1)
        # print(ana[:10,:10])
        # print(num[:10,:10])
