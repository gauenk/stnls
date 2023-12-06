
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
SAVE_DIR = Path("./output/tests/non_local_gather")

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def pytest_generate_tests(metafunc):
    test_lists = {"nheads":[1],"ws":[9],"wt":[1],
                  "stride0":[1],"stride1":[1],
                  "full_ws":[True],"seed":[123]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)


def test_forward(nheads,ws,wt,stride0,stride1,full_ws,seed):
    """

    Test the CUDA code with torch code

    Forward Pass

    """

    # -- get args --
    device = "cuda:0"
    set_seed(seed)
    W_t = 2*wt+1
    B,HD,T,F,H,W = 1,nheads,3,3,32,32
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    K = ws*ws*W_t
    itype = "int"

    # -- load flows --
    flows = th.ones((B,HD,T,W_t-1,2,nH,nW)).cuda()/2.
    flows = th.rand_like(flows)/2.+th.randint_like(flows,-3,3)+0.2
    flows = th.zeros_like(flows)
    flows = flows.round().int()

    # -- load top-k flows --
    ps = 1
    search = stnls.search.NonLocalSearch(ws, wt, ps, K, nheads, dist_type="l2",
                                         stride0=stride0, stride1=stride1,
                                         reflect_bounds=True,self_action="anchor",
                                         itype="int",full_ws=full_ws)
    vid = th.rand((B,HD,T,F,H,W)).to(device)
    dists,flows_k = search(vid,vid,flows)

    # -- gather --
    agg = stnls.agg.NonLocalGather(ps=ps,stride0=stride0,itype="int")
    stack = agg(vid,s_weight,s_flows_k)
    stack = rearrange(stack,'b hd t c h w -> b t (hd c) h w')

    # -- scatter --
    agg = stnls.agg.NonLocalScatterAdd(ps=ps,stride0=stride0,itype="int")
    stack = agg(vid,s_weight,s_flows_k)
    stack = rearrange(stac,'b hd t c h w -> b t (hd c) h w')


