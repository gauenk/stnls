"""
Incomplete.

"""

# -- python --
import sys

# -- data mgnmt --
from pathlib import Path
from easydict import EasyDict as edict

# -- testing --
import pytest

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- dnls --
import dnls

# -- meshgrid --


# -- test func --
from torch.nn.functional import fold,unfold,pad
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/")

def print_gpu_stats(gpu_bool,note=""):
    if gpu_bool:
        gpu_max = th.cuda.memory_allocated()/(1024**3)
        print("[%s] GPU Max: %2.4f" % (note,gpu_max))

def pytest_generate_tests(metafunc):
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)
    # test_lists = {"ps":[3],"stride":[1],"dilation":[1,2],
    #               "top":[3],"btm":[62],"left":[2],"right":[62]}
    test_lists = {"ps":[4],"stride":[1,2],"dilation":[2],
                  "top":[4],"btm":[64],"left":[1],"right":[61]}
    # test_lists = {"ps":[3,4,5,6,7,8],"stride":[1,2,3,4,5],"dilation":[1,2,3,4,5],
    #               "top":[3],"btm":[57],"left":[7],"right":[57]}
    # test_lists = {"ps":[3],"stride":[2],"dilation":[2],
    #               "top":[3],"btm":[57],"left":[7],"right":[57]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

#
# -- Test Against Batching Fold/Unfold --
#

def test_nn(ps,stride,dilation):
    return

    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    chnls,k,pt = 1,1,1
    ws,wt = 10,0
    adj = True
    top,btm,left,right = 0,64,0,64 # full image

    # -- sub square --
    coords = [top,left,btm,right]
    sq_h = coords[2] - coords[0]
    sq_w = coords[3] - coords[1]

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    exact = True

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device).contiguous()
    vid = th.ones_like(vid)

    # -- compute optical flow --
    flow = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- image params --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    nframes,height,width = t,h,w
    vshape = vid.shape

    # -- num of steps each direction --
    npix = t * h * w
    n_h = (sq_h - (ps-1)*dil - 1)//stride + 1
    n_w = (sq_w - (ps-1)*dil - 1)//stride + 1

    # -- skip if invalid shape --
    valid_h = (sq_h - (ps-1)*dil - 1) % stride == 0
    valid_w = (sq_w - (ps-1)*dil - 1) % stride == 0
    valid = valid_h and valid_w
    if not(valid):
        print("invalid: ",ps,dil,stride,coords)

    #
    # -- test logic --
    #
    unfold = dnls.iUnfold(ps,coords,stride=stride,dilation=dil,adj=adj)
    fold = dnls.iFold(vshape,coords,stride=stride,dilation=dil,adj=adj)

    # -- run through --
    patches_nl = unfold(vid,0)
    vid_nl = fold(patches_nl,0)

    # -- test --
    error = th.sum((vid_nl - vid)**2).item()
    assert error < 1e-10

    # -- clean-up --
    th.cuda.empty_cache()
    del vid,flow
    del vid_nn,vid_nl
    del patches_nl,patches_nn
    del grad_nn,grad_nl,vid_grad
    th.cuda.empty_cache()
