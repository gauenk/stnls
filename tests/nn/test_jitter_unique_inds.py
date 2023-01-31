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
    test_lists = {"ps":[7],"stride":[4],"dilation":[1],"wt":[0],
                  "ws":[-1,8],"top":[0],"btm":[64],"left":[0],"right":[64],"k":[-1,5],
                  "exact":[True],"seed":[123]}
    # test_lists = {"ps":[3,4,5,6,7,8],"stride":[1,2,3,4,5,8],"dilation":[1,2,3,4,5,8],
    #               "top":[1,11],"btm":[50,57],"left":[3,7],"right":[57,30]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def test_fwd(ps,stride,dilation,exact):
    """

    Test the CUDA code with torch code

    Forward Pass

    """

    #
    # -- get args --
    #

    # -- data params --
    B = 1
    T = 5
    F = 3
    H = 256
    W = 256

    # -- interp params --
    stride0 = 4
    scale = 2

    # -- search args --
    ws = 21
    wt = 3
    ps = 7
    K = 10

    # -- init data --
    vid0 = th.randn((B,T,F,H,W)).to("cuda:0")
    vid1 = th.randn((B,T,F,H,W)).to("cuda:0")
    fflow = th.randn((B,T,2,H,W)).to("cuda:0")
    bflow = th.randn((B,T,2,H,W)).to("cuda:0")

    # -- run search --
    dists,inds = dnls.search.nls(vid0,vid1,fflow,bflow,
                                 ws,wt,ps,K,stride0=stride0)
    inds_search = inds.clone()

    # -- upsampling --
    inds = dnls.nn.interpolate_inds(inds,scale,stride0,T,H,W)
    inds_interp = inds.clone()

    # -- ensure dups --
    dups,any_dup = dnls.testing.find_duplicate_inds(inds)
    print(th.sum(dups))
    assert any_dup, "Want duplicates for test."

    # -- jittering --
    inds = dnls.nn.jitter_unique_inds(inds,3,K,H,W)

    # -- check delta --
    args = th.where(th.abs(inds_interp - inds)>0)
    if len(args[0]) > 0:
        print(inds_interp[0,0,args[2][0]])
        print(inds[0,0,args[2][0]])

    # -- ensure no dups --
    dups,any_dup = dnls.testing.find_duplicate_inds(inds)
    print(th.sum(dups))

    # -- info --
    args = th.where(dups == True)
    if len(args[0]) > 0:
        scale2 = scale*scale
        loc = args[2][0]
        print(loc)
        print(inds.shape,dups.shape)
        # print(inds_tmp[0,0,args[2][0]//scale-1])
        print(inds_search[0,0,args[2][0]//scale])
        print(inds_interp[0,0,args[2][0]])
        # print(inds_tmp[0,0,args[2][0]//scale])
        print(inds[0,0,args[2][0]])
        print(inds[0,0,args[2][0]] - inds_interp[0,0,args[2][0]])
        # print(dists[0,0,args[2][0]])
        # print(dups[0,0,args[2][0]])
        # print(inds_tmp[0,0,args[2][0]])
        # print(dists_tmp[0,0,args[2][0]])

    # -- test --
    assert not(any_dup),"No dups!"
