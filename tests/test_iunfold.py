
# -- python --
import sys

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
    test_lists = {"ps":[3,7,11],"stride":[1,2,3,4,5],"dilation":[1,2,3,4,5],
                  "top":[3,11],"btm":[50,57],"left":[3,7],"right":[57,50]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

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
    top,left,btm,right = 2,2,h-12,w-12
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
    iunfold_nl = dnls.iunfold.iUnfold(ps,coords,stride=stride,dilation=dil)

    #
    # -- test logic --
    #

    # -- prepare videos --
    vid_nn = vid.clone()
    vid_nl = vid.clone()
    vid_nl.requires_grad_(True)

    # -- prepare video with boarder --
    padp = dil*(ps//2)
    vid_nn_cc = vid_nn[:,:,top:btm,left:right]#.contiguous()
    vid_nn = pad(vid_nn_cc,[padp,]*4,mode="reflect")
    vid_nn.requires_grad_(True)

    # -- run forward --
    patches_nn = run_unfold(vid_nn,ps,stride,dil)
    patches_nl = iunfold_nl(vid_nl,0,qTotal)

    # -- test cropped region v.s. our folded --
    # viz_nl,w_nl = run_fold(patches_nl,t,sq_h,sq_w,stride,dil)
    # viz_nl /= w_nl
    # if real_diff.max() > 1e-6:
    #     real_diff /= real_diff.max()
    # dnls.testing.data.save_burst(real_diff,SAVE_DIR,"real_diff")
    # real_diff = th.sum(real_diff).item()
    # assert real_diff < 1e-10

    # -- reshape and grad --
    shape_str = '(t h w) 1 1 c ph pw -> t h w c ph pw'
    patches_nn = rearrange(patches_nn,shape_str,t=t,h=n_h)
    patches_nl = rearrange(patches_nl,shape_str,t=t,h=n_h)
    patches_grad = th.rand_like(patches_nn).type(th.float32)

    # -- backward --
    th.autograd.backward(patches_nn,patches_grad)
    th.autograd.backward(patches_nl,patches_grad)

    # -- get grads --
    grad_nn = center_crop(vid_nn.grad,(sq_h,sq_w))
    grad_nl = vid_nl.grad[:,:,top:btm,left:right]

    # -- check forward --
    error = th.sum((patches_nn - patches_nl)**2).item()
    assert error < 1e-10

    # -- compute error --
    diff = th.abs(grad_nn - grad_nl)
    dmax = diff.max()
    if dmax > 1e-3: diff /= dmax
    dnls.testing.data.save_burst(diff,SAVE_DIR,"diff")

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
    top,left,btm,right = 2,2,h-12,w-12
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
    iunfold_nl = dnls.iunfold.iUnfold(ps,coords,stride=stride,dilation=dil)

    # -- prepare videos --
    vid_nl = vid.clone()
    vid_nl = vid_nl.requires_grad_(True)

    # -- prepare video with boarder --
    padp = dil*(ps//2)
    vid_nn = vid.clone()
    vid_nn = vid_nn[:,:,top:btm,left:right]
    vid_nn = pad(vid_nn,[padp,]*4,mode="reflect")
    vid_nn.requires_grad_(True)

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
    patches_nn = run_unfold(vid_nn,ps,stride,dil)

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
    grad_nn = center_crop(vid_nn.grad,(sq_h,sq_w))
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

    # -- unpack --
    ps = patches.shape[-1]
    padf = dil * (ps//2)
    hp,wp = h+2*padf,w+2*padf
    shape_str = '(t np) 1 1 c h w -> t (c h w) np'
    patches = rearrange(patches,shape_str,t=t)
    ones = th.ones_like(patches)

    # -- folded --
    vid_pad = fold(patches,(hp,wp),(ps,ps),stride=stride,dilation=dil)
    vid = center_crop(vid_pad,(h,w))

    # -- weigthed vid --
    wvid_pad = fold(ones,(hp,wp),(ps,ps),stride=stride,dilation=dil)
    wvid = center_crop(wvid_pad,(h,w))

    return vid,wvid

def run_unfold(_vid,_ps,_stride=1,_dil=1):

    # -- avoid fixutres --
    vid,stride = _vid,_stride
    ps,dil = _ps,_dil

    # -- run --
    shape_str = 't (c h w) np -> (t np) 1 1 c h w'
    patches = unfold(vid,(ps,ps),stride=stride,dilation=dil)
    patches = rearrange(patches,shape_str,h=ps,w=ps)
    return patches


# def run_fold(patches,t,h,w,stride=1,dil=1,ones=None):
#     ps = patches.shape[-1]
#     psHalf = ps//2
#     padf = dil * psHalf
#     hp,wp = h+2*padf,w+2*padf
#     shape_str = '(t np) 1 1 c h w -> t (c h w) np'
#     patches = rearrange(patches,shape_str,t=t)
#     if ones is None:
#         ones = th.ones_like(patches)
#     vid_pad = fold(patches,(hp,wp),(ps,ps),stride=stride,dilation=dil)
#     vid = center_crop(vid_pad,(h,w))
#     wvid_pad = fold(ones,(hp,wp),(ps,ps),stride=stride,dilation=dil)
#     wvid = center_crop(wvid_pad,(h,w))
#     return vid,wvid

# def run_unfold(vid_pad,ps,stride=1,dil=1):
#     # psHalf = ps//2
#     # vid_pad = pad(vid,4*[psHalf,],mode="reflect")
#     shape_str = 't (c h w) np -> (t np) 1 1 c h w'
#     patches = unfold(vid_pad,(ps,ps),stride=stride,dilation=dil)
#     patches = rearrange(patches,shape_str,h=ps,w=ps)
#     return patches

