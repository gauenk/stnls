

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
from torch.nn.functional import softmax
from torch.nn.functional import fold,unfold,pad
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/nn/non_local_attn/")

def pytest_generate_tests(metafunc):
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)
    test_lists = {"ws":[5,21],"wt":[0,1,3],
                  "stride0":[1,4],"stride1":[1,4],
                  "full_ws":[True],"seed":[0]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def test_use_adj():

    # -- get args --
    seed = 123
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    device = "cuda:0"
    run_flow = False
    T = 5
    k = 10
    wt = (T-1)//2
    pt = 1
    ws = 3
    stride0 = 1
    ps = 5
    set_seed(seed)

    # -- load data --
    vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid[0] = vid[0] + th.randn_like(vid[0])
    vid = vid.to(device)[:,:T,].contiguous()
    vid = repeat(vid,'b t c h w -> b t (r c) h w',r=12)[:,:32].contiguous()
    # vid = vid.to(device)[:,:1,].contiguous()
    # vid = repeat(vid,'b 1 c h w -> b t (r c) h w',t=T,r=12)[:,:32].contiguous()
    vid -= vid.min()
    vid /= vid.max()
    C = vid.shape[-3]

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(run_flow,False,vid,vid,0.)
    flows.fflow = 10*th.zeros_like(flows.fflow)
    flows.bflow = 10*th.zeros_like(flows.bflow)

    for use_adj in [True,False]:

        # -- search --
        print(vid.min(),vid.max())
        search = stnls.search.NonLocalSearch(ws, wt, ps, k, stride0=stride0,
                                             dist_type="l2",use_adj=use_adj,
                                             anchor_self=True)
        dists,inds = search(vid,vid,flows.fflow,flows.bflow)
        dists = dists.contiguous()#/(C*ps**2)
        inds = inds.contiguous()
        print(th.mean(th.sqrt(dists),dim=(0,1,2)))

        # -- weights --
        weights = softmax(-dists*10,dim=-1)
        print(th.mean(weights,dim=(0,1,2)))

        # -- aggregate patches --
        wpsum = stnls.reducer.WeightedPatchSum(ps, use_adj=use_adj)

        # -- fold into video --
        fold = stnls.iFoldz(vid.shape,stride=stride0,
                            use_adj=use_adj,device=vid.device)

        # -- exec --
        wpatches = wpsum(vid,weights,inds)#.view(scores_s.shape[0],-1)
        wpatches = rearrange(wpatches,'b H q pt c h w -> b q 1 pt (H c) h w')
        vidw,vidz = fold(wpatches)
        vidw = vidw / vidz

        # -- error --
        error = th.abs(vidw - vid)/(th.abs(vid)+1e-5)
        error = th.mean(error[th.where(th.abs(vid) > 1e-2)]).item()

        # -- check --
        print(use_adj,error)
        # if use_adj:
        #     assert error > 1e-1
        # else:
        #     assert error < 1e-5



