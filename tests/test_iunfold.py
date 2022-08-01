
# -- python --
import sys,pytest

# -- data mgnmt --
from pathlib import Path
from easydict import EasyDict as edict

# -- testing --
import unittest

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- dnls --
import dnls

# -- test func --
import torch.nn.functional as nnf
from torch.nn.functional import fold,unfold,pad
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/")

#
# -- Primary Testing Class --
#

def pytest_generate_tests(metafunc):
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)
    # test_lists = {"ps":[3],"stride":[6],"dilation":[1,2,3,4,5,6],
    #               "top":[0],"btm":[64],"left":[0],"right":[64]}
    # test_lists = {"ps":[3,4,5,6,7,8],"stride":[1,2,3,4,5,6,7,8],
    #               "dilation":[1,2,3,4,5,6],
    #               "top":[0],"btm":[64],"left":[0],"right":[64]}
    test_lists = {"ps":[3,4,7,8],"stride":[1,2,3,4,5,8],"dilation":[1,2,3,4,5,8],
                  "top":[0],"btm":[64],"left":[0],"right":[64]}
    # test_lists = {"ps":[3,4,7,8,9],"stride":[1,2,3,4,5,8],"dilation":[1,2,3,4,5,8],
    #               "top":[3,11],"btm":[50,57],"left":[3,7],"right":[57,50]}
    # test_lists = {"ps":[3,4,7,8,9,11],"stride":[1,2,3,4,5,8],"dilation":[1,2,3,4,5,8],
    #               "top":[3,11],"btm":[50,57],"left":[3,7],"right":[57,50]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def test_nn_with_fold(ps,stride,dilation):
    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    chnls,k,pt = 1,1,1
    ws,wt = 10,0
    top,btm,left,right = 0,128,0,128

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    exact = True

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext="jpg")
    vid = th.from_numpy(vid).to(device).contiguous()

    # -- make vid bigger --
    vid = th.cat([vid,vid],-1)
    vid = th.cat([vid,vid],-2)

    # -- compute flow --
    flow = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image info --
    device = vid.device
    shape = vid.shape
    t,c,h,w = shape
    npix = t * h * w

    # -- sub square --
    coords = [top,left,btm,right]
    sq_h = coords[2] - coords[0]
    sq_w = coords[3] - coords[1]

    # -- num of steps each direction --
    nframes = t
    npix = t * h * w
    n_h = (sq_h - (ps-1)*dil - 1)//stride + 1
    n_w = (sq_w - (ps-1)*dil - 1)//stride + 1

    # -- batching info --
    qTotal = t * n_h * n_w
    qSize = qTotal
    nbatches = (qTotal-1) // qSize + 1
    vid = vid.contiguous()

    # -- exec iunfold fxns --
    iunfold_nl = dnls.iUnfold(ps,coords,stride=stride,dilation=dil,
                              match_nn=True)
    # adj=True,only_full=True)

    #
    # -- test logic --
    #

    # -- prepare videos --
    vid_nl = vid.clone()
    vid_nl.requires_grad_(True)
    vid_nn = vid.clone()
    vid_nn.requires_grad_(True)

    # -- run forward --
    patches_nn = run_unfold(vid_nn,None,ps,stride,dil)
    patches_nl = iunfold_nl(vid_nl,0,qTotal)

    # -- reshape and grad --
    shape_str = '(t h w) 1 1 c ph pw -> t h w c ph pw'
    patches_nn = rearrange(patches_nn,shape_str,t=t,h=n_h)
    patches_nl = rearrange(patches_nl,shape_str,t=t,h=n_h)
    patches_grad = th.rand_like(patches_nn).type(th.float32)

    # -- backward --
    th.autograd.backward(patches_nn,patches_grad)
    th.autograd.backward(patches_nl,patches_grad)

    # -- get grads --
    # grad_nn = crop_pads(vid_nn.grad,sq_h,sq_w,ps,dil)
    grad_nn = vid_nn.grad
    grad_nl = vid_nl.grad#[:,:,top:btm,left:right]

    # -- compute error --
    diff = th.abs(grad_nn - grad_nl)
    dmax = diff.max()
    if dmax > 1e-3: diff /= dmax
    dnls.testing.data.save_burst(diff,SAVE_DIR,"diff")

    # -- check forward --
    # print("-"*20)
    # print(patches_nn[0,8,8,0])
    # print(patches_nl[0,9,9,0])
    # print("-"*20)
    # print(patches_nl[0,0,0,0])
    # print(patches_nl[0,1,1,0])
    # print(patches_nn[0,0,0,0])
    # print("-"*20)
    error = th.sum((patches_nn - patches_nl)**2).item()
    assert error < 1e-10

    # -- test backward --
    error = th.sum((grad_nn - grad_nl)**2).item()
    assert error/nframes < 1e-5

    # -- cleanup --
    th.cuda.empty_cache()
    del vid,flow
    del vid_nn,vid_nl
    del patches_nl,patches_nn
    del grad_nn,grad_nl,patches_grad
    th.cuda.empty_cache()


def test_nn(ps,stride,dilation,top,btm,left,right):

    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    chnls,k,pt = 1,1,1
    ws,wt = 10,0

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    exact = True

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext="jpg")
    vid = th.from_numpy(vid).to(device).contiguous()
    flow = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image info --
    device = vid.device
    shape = vid.shape
    t,c,h,w = shape
    npix = t * h * w

    # -- sub square --
    coords = [top,left,btm,right]
    sq_h = coords[2] - coords[0]
    sq_w = coords[3] - coords[1]

    # -- batching info --
    n_h = (sq_h-1)//stride+1
    n_w = (sq_w-1)//stride+1
    qTotal = t * n_h * n_w
    qSize = qTotal
    nbatches = (qTotal-1) // qSize + 1
    vid = vid.contiguous()

    # -- exec iunfold fxns --
    iunfold_nl = dnls.iUnfold(ps,coords,stride=stride,dilation=dil)

    #
    # -- test logic --
    #

    # -- prepare videos --
    vid_nl = vid.clone()
    vid_nl.requires_grad_(True)

    # -- pad nn video to allow for non-refl boundary --
    vid_nn = vid.clone()
    vid_nn = pad_video(vid_nn,coords,ps,stride,dil)
    vid_nn.requires_grad_(True)

    # -- run forward --
    patches_nn = run_unfold(vid_nn,None,ps,stride,dil)
    patches_nl = iunfold_nl(vid_nl,0,qTotal)

    # -- reshape and grad --
    shape_str = '(t h w) 1 1 c ph pw -> t h w c ph pw'
    patches_nn = rearrange(patches_nn,shape_str,t=t,h=n_h)
    patches_nl = rearrange(patches_nl,shape_str,t=t,h=n_h)
    patches_grad = th.rand_like(patches_nn).type(th.float32)

    # -- backward --
    th.autograd.backward(patches_nn,patches_grad)
    th.autograd.backward(patches_nl,patches_grad)

    # -- get grads --
    grad_nn = crop_pads(vid_nn.grad,sq_h,sq_w,ps,dil)
    grad_nl = vid_nl.grad[:,:,top:btm,left:right]

    # -- check forward --
    error = th.sum((patches_nn - patches_nl)**2).item()
    assert error < 1e-10

    # -- compute error --
    # diff = th.abs(grad_nn - grad_nl)
    # dmax = diff.max()
    # if dmax > 1e-3: diff /= dmax
    # dnls.testing.data.save_burst(diff,SAVE_DIR,"diff")

    # -- test backward --
    error = th.sum((grad_nn - grad_nl)**2).item()
    assert error < 1e-5

    # -- cleanup --
    th.cuda.empty_cache()
    del vid,flow
    del vid_nn,vid_nl
    del patches_nl,patches_nn
    del grad_nn,grad_nl,patches_grad
    th.cuda.empty_cache()


# @pytest.mark.skip(reason="too long right now")
def test_batched(ps,stride,dilation,top,btm,left,right):

    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    chnls,k,pt = 1,1,1
    ws,wt = 10,0

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    exact = True

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext="jpg")
    vid = th.from_numpy(vid).to(device).contiguous()
    flow = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image info --
    device = vid.device
    shape = vid.shape
    t,c,h,w = shape
    npix = t * h * w

    # -- sub square --
    coords = [top,left,btm,right]
    sq_h = coords[2] - coords[0]
    sq_w = coords[3] - coords[1]

    # -- batching info --
    npix = t * h * w
    n_h = (sq_h-1)//stride+1
    n_w = (sq_w-1)//stride+1
    qSize = 128
    qTotal = t * n_h * n_w
    nbatches = (qTotal-1) // qSize + 1

    # -- functions --
    iunfold_nl = dnls.iUnfold(ps,coords,stride=stride,dilation=dil)

    # -- prepare videos --
    vid_nl = vid.clone()
    vid_nl = vid_nl.requires_grad_(True)

    # -- pad nn video to allow for non-refl boundary --
    vid_nn = vid.clone()
    vid_nn = pad_video(vid_nn,coords,ps,stride,dil)
    vid_nn.requires_grad_(True)
    # padp = dil*(ps//2)
    # vid_nn = vid.clone()
    # vid_nn = vid_nn[:,:,top:btm,left:right]
    # vid_nn = pad(vid_nn,[padp,]*4,mode="reflect")
    # vid_nn.requires_grad_(True)

    # -- exec forward nl --
    patches_nl = []
    for index in range(nbatches):

        # -- get batch info --
        qindex = min(qSize * index,qTotal)
        qSize = min(qSize,qTotal-qindex)

        # -- run forward --
        patches_nl_i = iunfold_nl(vid_nl,qindex,qSize)

        # -- agg for testing --
        patches_nl.append(patches_nl_i)

    # -- cat for testing --
    patches_nl = th.cat(patches_nl,0)

    # -- exec forward nn --
    patches_nn = run_unfold(vid_nn,None,ps,stride,dil)

    # -- exec backward --
    shape_str = '(t h w) 1 1 c ph pw -> t h w c ph pw'
    patches_nn = rearrange(patches_nn,shape_str,t=t,h=n_h)
    patches_nl = rearrange(patches_nl,shape_str,t=t,h=n_h)
    patches_grad = th.rand_like(patches_nn).type(th.float32)
    th.autograd.backward(patches_nn,patches_grad)
    th.autograd.backward(patches_nl,patches_grad)

    # -- check forward --
    error = th.sum((patches_nn - patches_nl)**2).item()
    assert error < 1e-10

    # -- get grads --
    grad_nn = crop_pads(vid_nn.grad,sq_h,sq_w,ps,dil)
    grad_nl = vid_nl.grad[:,:,top:btm,left:right]

    # -- check backward --
    error = th.sum((grad_nn - grad_nl)**2).item()
    assert error < 1e-5

    # -- cleanup --
    th.cuda.empty_cache()
    del vid,flow
    del vid_nn,vid_nl
    del patches_nl,patches_nn
    del grad_nn,grad_nl,patches_grad
    th.cuda.empty_cache()


def run_fold(_patches,_t,_h,_w,_stride=1,_dil=1):
    # -- avoid pytest fixtures --
    patches = _patches
    t,h,w = _t,_h,_w
    stride,dil = _stride,_dil

    # -- folded --
    vid = fold(patches,(h,w),(ps,ps),stride=stride,dilation=dil)

    # -- weigthed vid --
    wvid = fold(ones,(h,w),(ps,ps),stride=stride,dilation=dil)

    return vid,wvid

def run_unfold(_vid,_pads,_ps,_stride=1,_dil=1):

    # -- avoid fixutres --
    vid,stride = _vid,_stride
    pads,ps,dil = _pads,_ps,_dil

    # -- run --
    shape_str = 't (c ph pw) hw -> (t hw) 1 1 c ph pw'
    patches = unfold(vid,(ps,ps),stride=stride,dilation=dil)
    patches = rearrange(patches,shape_str,ph=ps,pw=ps)
    return patches

def crop_pads(vid,h,w,ps,dil):
    pad_lg,pad_sm = dil*(ps//2),dil*((ps-1)//2)
    return vid[...,pad_lg:pad_lg+h,pad_lg:pad_lg+w]

def pad_video(vid,coords,ps,stride,dil):
    #
    # -- add video boarder for no edge effects from coords --
    #

    # -- compute pads --
    pad_lg,pad_sm = dil*(ps//2),dil*((ps-1)//2)
    t,c,h,w = vid.shape
    pcoords = padded_coords(coords,h,w,pad_lg,pad_sm)
    pads = padded_boarder(coords,pcoords,pad_lg,pad_sm)
    top,left,btm,right = pcoords
    # -- include non-refl. boundary if possible --
    vid_cc = vid[:,:,top:btm,left:right]
    # -- reflect to include ps//2 around edges if needed --
    vid_pad = pad(vid_cc,pads,mode="reflect")
    return vid_pad

def padded_coords(coords,h,w,pad_lg,pad_sm):
    top,left,btm,right = coords
    top_p = max(top-pad_lg,0)
    left_p = max(left-pad_lg,0)
    btm_p = min(btm+pad_sm,h)
    right_p = min(right+pad_sm,w)
    pcoords = [top_p,left_p,btm_p,right_p]
    return pcoords

def padded_boarder(coords,pcoords,pad_lg,pad_sm):
    top,left,btm,right = coords
    top_p,left_p,btm_p,right_p = pcoords
    top_b = pad_lg - (top - top_p)
    left_b = pad_lg - (left - left_p)
    btm_b = pad_sm - (btm_p - btm)
    right_b = pad_sm - (right_p - right)
    pads = [left_b,right_b,top_b,btm_b]
    return pads

