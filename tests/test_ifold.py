
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

# -- meshgrid --
import cache_io

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
    # test_lists = {"ps":[3],"stride":[3],"dilation":[2],
    #               "top":[11],"btm":[50],"left":[7],"right":[57]}
    test_lists = {"ps":[3,5,7,11],"stride":[1,2,3,4,5],"dilation":[1,2,3,4,5],
                  "top":[0,11],"btm":[64,50],"left":[0,7],"right":[64,57]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)
#
# -- Test Against Pytorch.nn.fold --
#

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
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device).contiguous()
    flow = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- image params --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    nframes,height,width = t,h,w
    vshape = vid.shape

    # -- sub square --
    coords = [top,left,btm,right]
    sq_h = coords[2] - coords[0]
    sq_w = coords[3] - coords[1]

    # -- batching info --
    npix = t * h * w
    nh = (sq_h-1)//stride+1
    nw = (sq_w-1)//stride+1
    qTotal = t * nh * nw
    qSize = qTotal
    nbatches = (qTotal-1) // qSize + 1

    # -- exec fold fxns --
    scatter_nl = dnls.scatter.ScatterNl(ps,pt,dilation=dil,exact=True)
    fold_nl = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dil)

    # -- patches for ifold --
    index = 0
    queryInds = dnls.utils.inds.get_iquery_batch(index,qSize,stride,
                                                 coords,t,h,w,device)
    nlDists,nlInds = dnls.simple.search.run(vid,queryInds,
                                            flow,k,ps,pt,ws,wt,chnls)
    assert th.sum(queryInds - nlInds[:,0]) < 1e-10
    patches_nl = scatter_nl(vid,nlInds)
    patches_nn = patches_nl.clone()
    patches_nn = patches_nn.requires_grad_(True)
    patches_nl = patches_nl.requires_grad_(True)

    #
    # -- test logic --
    #

    # -- run forward --
    top,left,btm,right = coords
    vid_nn,_ = run_fold(patches_nn,t,sq_h,sq_w,stride,dil)
    vid_nl = fold_nl(patches_nl,0)[:,:,top:btm,left:right]

    # -- run backward --
    vid_grad = th.randn_like(vid_nl)
    th.autograd.backward(vid_nn,vid_grad)
    th.autograd.backward(vid_nl,vid_grad)

    # -- check forward --
    delta = vid_nn - vid_nl
    error = th.sum(delta**2).item()
    assert error < 1e-10

    # -- check backward --
    grad_nn = patches_nn.grad
    grad_nl = patches_nl.grad

    # -- rearrange --
    shape_str = '(t h w) 1 1 c ph pw -> t c h w ph pw'
    grad_nn = rearrange(grad_nn,shape_str,t=t,h=nh)
    grad_nl = rearrange(grad_nl,shape_str,t=t,h=nh)

    # -- check backward --
    error = th.sum((grad_nn - grad_nl)**2).item()
    assert error < 1e-10

    # -- clean-up --
    th.cuda.empty_cache()
    del vid,flow
    del vid_nn,vid_nl
    del patches_nl,patches_nn
    del grad_nn,grad_nl,vid_grad
    del queryInds,nlDists,nlInds
    th.cuda.empty_cache()

#
# -- Test a Batched Ours Against Pytorch.nn.fold --
#

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
    exact = True
    gpu_stats = False

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device).contiguous()
    flow = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)
    print_gpu_stats(gpu_stats,"post-io")

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape

    # -- sub square --
    coords = [top,left,btm,right]
    sq_h = coords[2] - coords[0]
    sq_w = coords[3] - coords[1]

    # -- batching info --
    npix = t * h * w
    nh = (sq_h-1)//stride+1
    nw = (sq_w-1)//stride+1
    qTotal = t * nh * nw
    qSize = 512
    nbatches = (qTotal-1) // qSize + 1

    # -- exec fold fxns --
    scatter_nl = dnls.scatter.ScatterNl(ps,pt,dilation=dil,exact=True)
    fold_nl = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dil)
    patches_nl = []
    print_gpu_stats(gpu_stats,"pre-loop")

    for index in range(nbatches):

        # -- batch info --
        qindex = min(qSize * index,npix)
        qSize =  min(qSize, qTotal - qindex)

        # -- get patches --
        queryInds = dnls.utils.inds.get_iquery_batch(qindex,qSize,stride,
                                                     coords,t,h,w,device)
        nlDists,nlInds = dnls.simple.search.run(vid,queryInds,
                                                flow,k,ps,pt,ws,wt,chnls)
        patches_nl_i = scatter_nl(vid,nlInds)
        del queryInds,nlDists,nlInds
        th.cuda.empty_cache()

        patches_nl_i = patches_nl_i.requires_grad_(True)

        # -- run forward --
        vid_nl = fold_nl(patches_nl_i,qindex)
        patches_nl.append(patches_nl_i)

    # -- vis --
    print_gpu_stats(gpu_stats,"post-loop")

    # -- forward all at once --
    index,qSize = 0,qTotal
    queryInds = dnls.utils.inds.get_iquery_batch(index,qSize,stride,
                                                 coords,t,h,w,device)
    nlDists,nlInds = dnls.simple.search.run(vid,queryInds,
                                            flow,k,ps,pt,ws,wt,chnls)
    patches_nn = scatter_nl(vid,nlInds)
    patches_nn.requires_grad_(True)
    print_gpu_stats(gpu_stats,"post-search")
    vid_nn,_ = run_fold(patches_nn,t,sq_h,sq_w,stride,dil)
    print_gpu_stats(gpu_stats,"post-fold")

    # -- run backward --
    top,left,btm,right = coords
    vid_grad = th.randn_like(vid_nn)
    vid_nl = fold_nl.vid[:,:,top:btm,left:right]
    th.autograd.backward(vid_nn,vid_grad)
    th.autograd.backward(vid_nl,vid_grad)
    print_gpu_stats(gpu_stats,"post-bkw")

    # -- get grads --
    grad_nn = patches_nn.grad
    grad_nl = th.cat([p_nl.grad for p_nl in patches_nl])

    # -- check forward --
    top,left,btm,right = coords
    error = th.sum((vid_nn - vid_nl)**2).item()
    assert error < 1e-10

    # -- reshape --
    shape_str = '(t h w) 1 1 c ph pw -> t c h w ph pw'
    grad_nn = rearrange(grad_nn,shape_str,t=t,h=nh)
    grad_nl = rearrange(grad_nl,shape_str,t=t,h=nh)

    # -- check backward --
    error = th.sum((grad_nn - grad_nl)**2).item()
    assert error < 1e-10

    # -- clean-up --
    th.cuda.empty_cache()
    del vid,flow
    del vid_nn,vid_nl
    del patches_nl,patches_nn
    del grad_nn,grad_nl,vid_grad
    del queryInds,nlDists,nlInds
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
    psHalf = ps//2
    padf = dil * psHalf
    shape_str = 't (c h w) np -> (t np) 1 1 c h w'
    vid_pad = pad(vid,4*[padf,],mode="reflect")
    patches = unfold(vid_pad,(ps,ps),stride=stride,dilation=dil)
    patches = rearrange(patches,shape_str,h=ps,w=ps)
    return patches


