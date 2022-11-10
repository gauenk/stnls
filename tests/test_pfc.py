"""

Testing for Patch Fully-Connected Layers

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

# -- torch --
import torch.nn as nn

# -- dnls --
import dnls
import dnls.utils.gpu_mem as gpu_mem
from dnls.pytorch.simple import pfc as pfc_gt

# -- meshgrid --


# -- test func --
from torch.nn.functional import fold,unfold,pad
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/")

def pytest_generate_tests(metafunc):
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)
    test_lists = {"ps":[7],"stride":[1],"dilation":[1]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def test_fwd(ps,stride):

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    exact = True
    dnames = ["davis_baseball_64x64",]
    ext = "jpg"

    # -- load data --
    vid = dnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)/255.
    vid = vid.to(device).contiguous()
    # vid[:,:,0,:,:] = vid[:,:,0,:,:]
    # vid[:,:,1,:,:] = vid[:,:,1,:,:]
    # vid[:,:,2,:,:] = vid[:,:,0,:,:]
    # vid[...] = 0.
    # vid[...,0,0] = 1.
    # vid[...,0,1] = 1.
    # vid[...,0,4] = 1.
    # vid[...,0,5] = 1.
    # vid[...,0,1] = 2.
    # vid[...,1,0] = 4.
    # vid[...,6,6] = 5.
    # vid[...,7,7] = 10.
    B,T,C,H,W = vid.shape
    c_in = C
    c_out = 5

    # -- compute optical flow --
    flow = dnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)

    # -- init fc layer --
    pfc = dnls.nn.init("pfc",c_in,c_out,ps,stride)
    dim1 = ps*ps*c_out
    dim0 = ps*ps*c_in
    fc_layer = nn.Linear(dim0,dim1).to(device)
    # print(fc_layer.weight.shape)
    fc_weight = th.rand_like(fc_layer.weight).to(device)
    # fc_weight[0,0] = 0.
    # fc_weight[0,1] = 2.
    # fc_weight[1,0] = 3.
    # fc_weight[1,1] = 4.
    # fc_weight[0,:] = th.arange(len(fc_weight[0,:]))
    # fc_weight[1,:] = th.arange(len(fc_weight[0,:]))
    fc_bias = th.rand_like(fc_layer.bias).to(device)
    fc_layer.weight.data = fc_weight
    fc_layer.bias.data = fc_bias
    pfc.weights = fc_weight
    pfc.bias = fc_bias

    # -- groundtruth --
    vid_gt = pfc_gt.run(vid,stride,ps,fc_layer,dil=1)

    # -- ours --
    vid_te = pfc(vid)

    # -- viz --
    # print("vid_gt.shape: ",vid_gt.shape)
    # print("-"*30)
    # print(vid_gt[0,0,0,:9,:9])
    # print(vid_te[0,0,0,:9,:9])
    # print("-"*30)
    # print(vid_gt[0,0,0,-9:,-9:])
    # print(vid_te[0,0,0,-9:,-9:])
    # print("-"*30)

    # print("-"*30)
    # args = th.where(th.abs(vid_gt - vid_te) > 1e-3)
    # print("-"*30)
    # print(args)
    # print(vid_gt[args])
    # print(vid_te[0,0,0,:3,:3])
    # print(vid_gt[0,0,:,:,3])
    # print(vid_te[0,0,:,:,3])
    # print(vid_gt[0,0,:,:,5])
    # print(vid_te[0,0,:,:,5])
    # print(vid_gt[0,0,:,-3:,-3:])
    # print(vid_te[0,0,:,-3:,-3:])

    # -- compare --
    diff = th.abs(vid_gt - vid_te)/(vid_gt.abs()+1e-5)
    diff = th.mean(diff).item()
    assert diff < 1e-5

def test_bwd():
    pass

