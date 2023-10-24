
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

# -- paths --
SAVE_DIR = Path("./output/tests/non_local_stack")

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def pytest_generate_tests(metafunc):
    test_lists = {"ps":[3],"stride0":[1,2],
                  "K":[10],"nheads":[1],
                  "seed":[0],"reflect_bounds":[False,True],
                  "itype":["float","int"]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)


def test_fwd(ps,stride0,K,nheads,reflect_bounds,itype,seed):

    """

    Test the CUDA code with torch code

    Forward Pass

    """

    # -- get args --
    pt = 1
    device = "cuda:0"
    set_seed(seed)

    # -- load data --
    B,HD,T,F,H,W = 1,nheads,2,1,8,8
    vid = th.rand((B,T,F*HD,H,W),device=device).float()

    # -- load weights --
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    weights = th.rand((B,HD,T,nH,nW,K)).to(device)
    # weights = th.ones_like(weights)
    # vid = th.ones_like(vid)

    # -- load flows --
    flows = th.ones((B,HD,T,nH,nW,K,3))+0.1
    flows = th.rand_like(flows)/2.+1.1
    tgrid = th.arange(0,T).view(1,1,T,1,1,1)
    flows[...,0] = th.randint(0,T,size=flows[...,0].shape)-tgrid
    flows[...,1:] = th.rand_like(flows[...,1:])/2.+0.2
    not_int = th.all(th.abs(flows[...,1:].round() - flows[...,1:])>1e-5).item()
    if itype == "float":
        assert not_int,"Gradcheck only works _not_ near an int."
    else:
        flows = flows.round().int()
    flows = flows.to(vid.device)

    # -- exec fold fxns --
    agg = stnls.agg.NonLocalStack(ps=ps,stride0=stride0,
                                  reflect_bounds=reflect_bounds,
                                  itype=itype)
    stack = agg(vid,weights,flows)
    stack_gt = stnls.testing.non_local_stack(vid,weights,flows,ps,stride0,
                                             reflect_bounds=reflect_bounds,itype=itype)

    assert th.allclose(stack,stack_gt,1e-2,1e-2,equal_nan=True)

def test_bwd(ps,stride0,K,nheads,reflect_bounds,itype,seed):

    """

    Test the CUDA code with torch code

    Forward Pass

    """

    # -- get args --
    pt = 1
    device = "cuda:0"
    set_seed(seed)

    # -- load data --
    B,HD,T,F,H,W = 1,nheads,2,3,8,8
    vid = th.rand((B,T,F*HD,H,W),device=device).float().requires_grad_(True)

    # -- load weights --
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    weights = th.rand((B,HD,T,nH,nW,K)).to(device).requires_grad_(True)

    # -- load flows --
    flows = th.ones((B,HD,T,nH,nW,K,3))+0.1
    flows = th.rand_like(flows)/2.+1.1
    tgrid = th.arange(0,T).view(1,1,T,1,1,1)
    flows[...,0] = th.randint(0,T,size=flows[...,0].shape)-tgrid
    flows[...,1:] = th.rand_like(flows[...,1:])/2.+2.2
    not_int = th.all(th.abs(flows[...,1:].round() - flows[...,1:])>1e-5).item()
    flows = flows.to(vid.device)
    flows_t,flows_sp = flows[...,[0]],flows[...,1:]
    if itype == "float":
        assert not_int,"Gradcheck only works _not_ near an int."
    else:
        flows = flows.round().int()

    # -- exec fold fxns --
    stacking = stnls.agg.NonLocalStack(ps=ps,stride0=stride0,
                                       reflect_bounds=reflect_bounds,
                                       itype=itype)
    stack = stacking(vid,weights,flows)

    # -- gradcheck --
    stack_weights = lambda weights: stacking(vid,weights,flows)
    th.autograd.gradcheck(stack_weights, weights, eps=1e-4,
                          atol=1e-2, nondet_tol=1e-7, raise_exception=True)
    stack_vid = lambda vid: stacking(vid,weights,flows)
    th.autograd.gradcheck(stack_vid, vid, eps=1e-4,
                          atol=1e-2, nondet_tol=1e-7, raise_exception=True)
    if itype == "float":
        flows_sp = flows_sp.requires_grad_(True)
        def stack_flows(flows_sp):
            flows = th.cat([flows_t,flows_sp],-1)
            return stacking(vid,weights,flows)
        th.autograd.gradcheck(stack_flows, flows_sp, eps=1e-2,
                              atol=1e-2, nondet_tol=1e-7, raise_exception=True)


