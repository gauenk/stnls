"""

Weighted-Patch (Inplace) Sum


Verbose Psuedo-Code:
   yi = softmax(dists)
   patches_i = scatter(b2,nlInds_cu).type(th.float64)
   patches_i = rearrange(patches_i,'n k 1 c h w -> n k (c h w)')
   zi = th.sum(yi * patches_i,1).type(th.float32) # n (c h w)

"""


# -- python --
import sys

# -- data mgnmt --
from pathlib import Path
from easydict import EasyDict as edict

# -- testing --
import unittest,pytest

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- dnls --
import dnls
import dnls.utils.gpu_mem as gpu_mem
from dnls.utils.pads import comp_pads

# -- meshgrid --
import cache_io

# -- test func --
from torch.nn.functional import fold,unfold,pad
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/")

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)

def pytest_generate_tests(metafunc):
    seed = 123
    set_seed(seed)
    # test_lists = {"ps":[3],"stride":[1],"dilation":[1,2],
    test_lists = {"ps":[7],"stride":[4],"dilation":[1],"wt":[3],
                  "ws":[10],"top":[0],"btm":[-1],"left":[0],"right":[-1]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)


def simple_run(vid,dists_s,inds,ps,pt,reflect_bounds,exact):
    scatter = dnls.scatter.ScatterNl(ps,pt,exact=exact,reflect_bounds=reflect_bounds)
    dists_s = dists_s[...,None].type(th.float64)
    patches_i = scatter(vid,inds).type(th.float64)
    patches_i = rearrange(patches_i,'n k 1 c h w -> n k (c h w)')
    print("0: ",patches_i[:3,0,:3])
    print("1: ",patches_i[:3,1,:3])
    wpatches_i = th.sum(dists_s * patches_i,1).type(th.float32)
    return wpatches_i

def test_forward(ps,stride,dilation,top,btm,left,right):

    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    chnls,pt = 1,1
    ws,wt,k = 10,5,2
    # ws,wt,k = -1,0,-1
    stride0 = stride
    stride1 = 1

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    exact = False
    gpu_stats = False
    adj = 0#ps//2
    reflect_bounds = True
    use_search_abs = ws == -1
    use_k = k != -1

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)/255.
    vid = th.from_numpy(vid).to(device)[:3,].contiguous()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],0)
    # vid = th.cat([vid,vid],0)
    print("vid.shape: ",vid.shape)

    # -- compute flow --
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape

    # -- sub square --
    top,btm,left,right = 0,h,0,w
    coords = [top,left,btm,right]

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid.shape, ps, stride0, dil)
    oh1,ow1,_,_ = comp_pads(vid.shape, ps, stride1, dil)
    n_h = (hp - (ps-1)*dil - 1)//stride0 + 1
    n_w = (wp - (ps-1)*dil - 1)//stride0 + 1

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h * n_w
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- init xsearch --
    xsearch = dnls.xsearch.CrossSearchNl(flows.fflow, flows.bflow, k, ps, pt,
                                         ws, wt, oh0, ow0, oh1, ow1,
                                         chnls=chnls,dilation=dil, stride=stride1,
                                         use_bound=reflect_bounds,
                                         use_k=use_k,use_search_abs=use_search_abs)

    # -- query inds --
    qindex = 0
    iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch,stride0,
                                                coords,t,device)

    # -- init our inner product --
    wpsum = dnls.wpsum.WeightedPatchSum(ps, pt, dilation=dil, reflect_bounds=reflect_bounds,
                                        adj=adj, exact=exact)

    # -- run search --
    dists,inds = xsearch(vid,iqueries,vid1=vid)
    dists_s = th.nn.functional.softmax(dists,1)

    # -- run simple for testing --
    wpatches_te = wpsum(vid,dists_s,inds).view(iqueries.shape[0],-1)
    wpatches_gt = simple_run(vid,dists_s,inds,ps,pt,reflect_bounds,exact)

    # -- compare --
    error = th.abs(wpatches_gt - wpatches_te).mean().item()
    assert error < 1e-10


def test_backward(ps,stride,dilation,top,btm,left,right):

    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    chnls,pt = 1,1
    ws,wt,k = 10,5,2
    # ws,wt,k = -1,0,-1
    stride0 = stride
    stride1 = 1

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    exact = True
    gpu_stats = False
    adj = 0#ps//2
    reflect_bounds = True
    use_search_abs = ws == -1
    use_k = k != -1

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)/255.
    vid = th.from_numpy(vid).to(device)[:3,].contiguous()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],0)
    # vid = th.cat([vid,vid],0)
    print("vid.shape: ",vid.shape)

    # -- compute flow --
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape

    # -- sub square --
    top,btm,left,right = 0,h,0,w
    coords = [top,left,btm,right]

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid.shape, ps, stride0, dil)
    oh1,ow1,_,_ = comp_pads(vid.shape, ps, stride1, dil)
    n_h = (hp - (ps-1)*dil - 1)//stride0 + 1
    n_w = (wp - (ps-1)*dil - 1)//stride0 + 1

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h * n_w
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- init xsearch --
    xsearch = dnls.xsearch.CrossSearchNl(flows.fflow, flows.bflow, k, ps, pt,
                                         ws, wt, oh0, ow0, oh1, ow1,
                                         chnls=chnls,dilation=dil, stride=stride1,
                                         use_bound=reflect_bounds,
                                         use_k=use_k,use_search_abs=use_search_abs)

    # -- query inds --
    qindex = 0
    iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch,stride0,
                                                coords,t,device)

    # -- init our inner product --
    wpsum = dnls.wpsum.WeightedPatchSum(ps, pt, dilation=dil, reflect_bounds=reflect_bounds,
                                        adj=adj, exact=exact)

    # -- run search --
    dists,inds = xsearch(vid,iqueries,vid1=vid)
    dists_s = th.nn.functional.softmax(dists,1)

    # -- prepare for grads --
    vid_te = vid.clone()
    vid_gt = vid.clone()
    vid_te.requires_grad_(True)
    vid_gt.requires_grad_(True)


    # -- forward passes --
    wpatches_te = wpsum(vid_te,dists_s,inds).view(iqueries.shape[0],-1)
    wpatches_gt = simple_run(vid_gt,dists_s,inds,ps,pt,reflect_bounds,exact)

    # -- backward passes --
    wpatches_grad = th.rand_like(wpatches_te)
    th.autograd.backward(wpatches_te,wpatches_grad)
    th.autograd.backward(wpatches_gt,wpatches_grad)

    # -- grab grads --
    grad_te = vid_te.grad
    grad_gt = vid_gt.grad
    print(grad_te[0,0,:3,:3])
    print(grad_gt[0,0,:3,:3])

    # -- compare --
    error = th.abs(grad_te - grad_gt).mean().item()
    assert error < 1e-10

