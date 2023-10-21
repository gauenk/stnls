
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

# -- testing functions --
from stnls.testing.non_local_stack_gt import non_local_stack_int
from stnls.testing.non_local_stack_gt import non_local_stack_bilin2d

# # -- test func --
# from torch.nn.functional import fold,unfold,pad
# from torchvision.transforms.functional import center_crop

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
    test_lists = {"ps":[3],"stride0":[3],"stride1":[1],
                  "dilation":[1],"wt":[1],"ws":[1],
                  "k":[0],"exact":[False],"nheads":[1],
                  "self_action":[None],"seed":[0,1,2,3],"dist_type":["prod"],
                  "k_agg":[-1]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)


def test_fwd(ws,wt,k,ps,stride0,stride1,dilation,k_agg,
             nheads,self_action,exact,dist_type,seed):

    """

    Test the CUDA code with torch code

    Forward Pass

    """

    # -- get args --
    pt = 1
    dil = dilation
    ext = "jpg"
    device = "cuda:0"
    clean_flow = True
    run_flow = False
    reflect_bounds = True
    use_adj = False # keep false since unfold/fold doesn't match search
    pt = 1
    device = "cuda:0"
    reflect_bounds = True
    set_seed(seed)
    itype = "float"

    # -- load data --
    K = 5
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
    stacking = stnls.agg.NonLocalStack(ps=ps,stride0=stride0,
                                       reflect_bounds=reflect_bounds,
                                       itype=itype)
    stack = stacking(vid,weights,flows)
    stack_gt = stnls.testing.non_local_stack(vid,weights,flows,ps,stride0,itype=itype)
    assert th.allclose(stack,stack_gt,1e-3,1e-3,equal_nan=True)

def test_bwd(ws,wt,k,ps,stride0,stride1,dilation,k_agg,
             nheads,self_action,exact,dist_type,seed):

    """

    Test the CUDA code with torch code

    Forward Pass

    """

    # -- get args --
    pt = 1
    dil = dilation
    device = "cuda:0"
    use_adj = False # keep false since unfold/fold doesn't match search
    pt = 1
    device = "cuda:0"
    reflect_bounds = True
    set_seed(seed)
    K = ws*ws if k <= 0 else k
    itype = "float"

    # -- load data --
    B,HD,T,F,H,W = 1,nheads,2,3,16,16
    vid = th.rand((B,T,F*HD,H,W),device=device).float().requires_grad_(True)

    # -- load weights --
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    weights = th.rand((B,HD,T,nH,nW,K)).to(device).requires_grad_(True)

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
    flows[...,1:] = flows[...,1:].requires_grad_(True)

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
        stack_flows = lambda flows: stacking(vid,weights,flows)
        th.autograd.gradcheck(stack_flows, flows, eps=1e-4,
                              atol=1e-2, nondet_tol=1e-7, raise_exception=True)


