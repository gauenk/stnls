
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
import stnls.utils.gpu_mem as gpu_mem
from stnls.utils.pads import comp_pads
from stnls.utils.inds import get_batching_info

# -- meshgrid --

# -- test func --
from torch.nn.functional import fold,unfold,pad
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/nn/non_local_inds/")

def pytest_generate_tests(metafunc):
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)
    test_lists = {"ws":[1],"wt":[1],
                  "stride0":[1],"stride1":[1],
                  "full_ws":[True],"seed":[0]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def test_fwd(ws,wt,stride0,stride1,full_ws,seed):
    """

    Test the CUDA code with torch code

    Forward Pass

    """

    # -- get args --
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    set_seed(seed)

    # -- init vars --
    device = "cuda:0"
    run_flow = False
    clean_flow = True
    reflect_bounds = True
    use_adj = False
    self_action = None
    pt,ps,k = 1,7,-1

    # -- load data --
    B,T,F,H,W = 2,10,16,16,16
    vid = th.ones((B,T,F,H,W),device=device).float()
    vid0 = th.randn_like(vid)-0.5
    vid1 = th.randn_like(vid)

    # -- load flows --
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    W_t = 2*wt+1
    flows = edict()
    flows.fflow = th.ones((B,T,2,nH,nW)).to(device)
    flows.bflow = th.ones((B,T,2,nH,nW)).to(device)
    flows.fflow = th.rand_like(flows.fflow)/(2*T)+1./(2*T)
    flows.bflow = th.rand_like(flows.bflow)/(2*T)+1./(2*T)

    # -- exec fold fxns --
    search = stnls.search.NonLocalSearch(ws, wt, ps, k,
                                         stride0=stride0, stride1=stride1,
                                         full_ws=full_ws, self_action=self_action)
    _,inds_gt = search(vid,vid,flows.fflow,flows.bflow)
    B,HD,T,nH,nW,K,_ = inds_gt.shape
    inds_gt = inds_gt.view(B*HD,T,nH,nW,K,3)
    print(inds_gt.shape)
    inds_gt = stnls.utils.misc.flow2inds(inds_gt,stride0)
    print(inds_gt.shape)
    inds_gt = inds_gt.view(B,HD,T*nH*nW,K,3)[:,0]
    # flow2inds(flow,stride0)

    # -- apply temporal inds --
    inds_te = stnls.nn.non_local_inds(flows.fflow,flows.bflow,ws,wt,
                                      stride0,stride1,full_ws)
    inds_te = inds_te.view(B,T*nH*nW,K,3)

    print(th.cat([inds_gt,inds_te],-1))

    # -- compare --
    diff = th.mean(th.mean(1.*(inds_te != inds_gt))).item()
    assert diff < 1e-10
