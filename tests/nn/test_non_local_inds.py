
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


def test_fwd(ws,wt,stride0,stride1,full_ws,seed):
    """

    Test the CUDA code with torch code

    Forward Pass

    """

    # -- get args --
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    pt = 1
    set_seed(seed)

    # -- init vars --
    device = "cuda:0"
    run_flow = False
    clean_flow = True
    reflect_bounds = True
    use_adj = False
    adj = 0
    rbwd = False

    # -- load data --
    vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid.to(device)[:,:5,].contiguous()
    vid = repeat(vid,'b t c h w -> b t (r c) h w',r=12)[:,:32].contiguous()
    vid /= vid.max()

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(run_flow,clean_flow,vid,vid,0.)
    flows.fflow = 10*th.randn_like(flows.fflow)
    flows.bflow = 10*th.randn_like(flows.bflow)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    B,T,color,H,W = shape
    vshape = vid.shape
    chnls = vid.shape[-3]
    # B,T,C,H,W = vid.shape

    # -- batching info --
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    ntotal = T * nH * nW
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1
    h0_off,w0_off = 0,0
    h1_off,w1_off = 0,0

    # -- exec fold fxns --
    ps,k = 7,-1
    search = stnls.search.NonLocalSearch(ws, wt, ps, k,
                                         stride0=stride0, stride1=stride1,
                                         full_ws=full_ws, anchor_self=False,
                                         remove_self=False)
    _,inds_gt = search(vid,vid,flows.fflow,flows.bflow)
    inds_gt = inds_gt[:,0]

    # -- apply temporal inds --
    inds_te = stnls.nn.non_local_inds(flows.fflow,flows.bflow,ws,wt,
                                      stride0,stride1,full_ws)
    B,Q,*_ = inds_te.shape
    inds_te = inds_te.reshape(B,Q,-1,3)

    # -- viz --
    # print(inds_gt.shape,inds_te.shape)
    # print("-"*20)
    # i = 0
    # print(th.cat([inds_gt[0][i],inds_te[0][i]],-1))
    # print(th.abs(inds_gt[0][i]-inds_te[0][i]))
    # print("-"*20)
    # print(inds_gt[0][100])
    # print(inds_te[0][100])
    # for i in range(inds_gt.shape[1]):
    #     eq = th.all(inds_gt[0][i] == inds_te[0][i]).item()
    #     print(i,eq)
    #     if not(eq): break

    # -- compare --
    diff = th.mean(th.mean(1.*(inds_te != inds_gt))).item()
    assert diff < 1e-10
