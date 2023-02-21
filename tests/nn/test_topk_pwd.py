# -- data mgnmt --
from pathlib import Path

# -- testing --
import pytest
import random

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- patchify --
from torch.nn.functional import fold,unfold,pad

# -- dnls --
import dnls
import dnls.utils.gpu_mem as gpu_mem
from dnls.utils.pads import comp_pads
from dnls.utils.inds import get_batching_info

# -- paths --
SAVE_DIR = Path("./output/tests/prod_search")

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def pytest_generate_tests(metafunc):
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)
    # test_lists = {"ps":[3],"stride":[1],"dilation":[1,2],
    #               "top":[3],"btm":[62],"left":[2],"right":[62]}
    # test_lists = {"ps":[4],"stride":[1,2],"dilation":[2],
    #               "top":[4],"btm":[64],"left":[1],"right":[61]}
    # test_lists = {"ps":[3,4,5,6,7,8],"stride":[1,2,3,4,5],"dilation":[1,2,3,4,5],
    #               "top":[3],"btm":[57],"left":[7],"right":[57]}
    # test_lists = {"ps":[3],"stride":[2],"dilation":[2],
    #               "top":[3],"btm":[57],"left":[7],"right":[57]}
    test_lists = {"ps":[7],"stride0":[4],"dilation":[1],"wt":[0],
                  "ws":[-1,8],"top":[0],"btm":[64],"left":[0],"right":[64],"k":[-1,5],
                  "exact":[True],"seed":[123]}
    # test_lists = {"ps":[3,4,5,6,7,8],"stride":[1,2,3,4,5,8],"dilation":[1,2,3,4,5,8],
    #               "top":[1,11],"btm":[50,57],"left":[3,7],"right":[57,30]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def test(ps,stride0,dilation,exact):
    """

    Test the CUDA code with torch code

    Forward Pass

    """

    # -- search info --
    ws = 10
    wt = 0
    pt = 1
    k = 10
    nheads = 1
    use_adj = False
    reflect_bounds = False

    # -- init data --
    B = 1
    HD = nheads
    T = 2
    C = 3
    H = 128
    W = 128
    vid = th.randn((B,HD,T,C,H,W),dtype=th.float32,device="cuda:0")
    vid1 = th.randn((B,HD,T,C,H,W),dtype=th.float32,device="cuda:0")
    vid2 = th.randn((B,HD,T,C,H,W),dtype=th.float32,device="cuda:0")
    zflow = th.zeros((B,T,2,H,W),dtype=th.float32,device="cuda:0")

    # -- get inds --
    search = dnls.search.NonLocalSearch(ws, wt, ps, k, nheads,
                                        dist_type="l2",dilation=dilation,
                                        stride0=stride0, use_adj=use_adj,
                                        reflect_bounds=reflect_bounds)
    _,inds0 = search(vid,vid1,zflow,zflow)
    _,inds1 = search(vid,vid1,zflow,zflow)


    # -- run test --
    pwd_te = dnls.nn.topk_pwd.run(vid,inds0,inds1,ps,
                                  pt,dilation,reflect_bounds,use_adj)
    pwd_gt = dnls.simple.topk_pwd.run(vid,inds0,inds1,ps,
                                      pt,dilation,reflect_bounds,use_adj)

    pwd_te = th.sort(pwd_te,-1)[0]
    pwd_gt = th.sort(pwd_gt,-1)[0]

    # -- compare --
    delta = th.mean((pwd_te - pwd_gt)**2).item()
    assert delta < 1e-5


