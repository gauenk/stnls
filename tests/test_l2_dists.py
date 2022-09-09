"""
Write "inds" check;
verify the correct meshgrid is included in the exhaustive search

see pacnet's test for details.


"""

# -- python --
import pytest
import numpy as np
import sys
from pathlib import Path
from easydict import EasyDict as edict

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- dnls --
import dnls
import dnls.utils.gpu_mem as gpu_mem
from dnls.utils.pads import comp_pads
from dnls.utils.inds import get_batching_info

# -- check if reordered --
SAVE_DIR = Path("./output/tests/l2_search_with_index")

#
# -- meshgrid --
#

def pytest_generate_tests(metafunc):
    seed = 234
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
    test_lists = {"ps":[7],"stride0":[4],"stride1":[4],"dilation":[1],"wt":[0],
                  "ws":[-1],"top":[0],"btm":[64],"left":[0],"right":[64],"k":[-1,10],
                  "exact":[True],"reflect_bounds":[False]}
    # test_lists = {"ps":[3,4,5,6,7,8],"stride":[1,2,3,4,5,8],"dilation":[1,2,3,4,5,8],
    #               "top":[1,11],"btm":[50,57],"left":[3,7],"right":[57,30]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)


#
# -- forward testing --
#

def test_cu_vs_th_fwd(k,ps,stride0,stride1,dilation,reflect_bounds,exact):
    """

    Test the CUDA code with torch code

    Forward Pass

    """
    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    pt = 1
    wt = 0
    ws = -1
    # stride0 = stride
    # stride1 = 1
    search_abs = True
    use_k = k>0
    exact = True

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    adj = True
    only_full = True

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device)[:1,].contiguous()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],-2)

    # -- normalize --
    vid /= vid.max()
    # vidr = th.ones_like(vid)
    vidr = th.rand_like(vid)

    # -- compute flow --
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape
    chnls = vid.shape[1]

    # -- unpack --
    _,_,n0,n1 = get_batching_info(vid.shape,stride0,stride1,ps,dil)
    n_h0,n_w0 = n0[0],n0[1]
    n_h1,n_w1 = n1[0],n1[1]

    # -- exec fold fxns --
    use_adj = True
    h0_off,w0_off,_,_ = comp_pads(vid.shape, ps, stride0, 1)
    h1_off,w1_off,_,_ = comp_pads(vid.shape, ps, stride1, 1)
    search = dnls.search.init("l2_with_index",
                              flows.fflow, flows.bflow, k, ps, pt,
                              ws, wt, dilation=dil,
                              stride0=stride0, stride1=stride1,
                              use_k = use_k,use_adj=use_adj,
                              reflect_bounds=reflect_bounds,
                              search_abs=search_abs,exact=exact,
                              h0_off=h0_off,w0_off=w0_off,
                              h1_off=h1_off,w1_off=w1_off)
    l2_dists = dnls.search.init("l2_dists", ps, pt,
                                dilation=dil,stride0=stride0,
                                use_adj=use_adj,
                                reflect_bounds=reflect_bounds,
                                exact=exact,
                                h0_off=h0_off,w0_off=w0_off,
                                h1_off=h1_off,w1_off=w1_off)

    # -- batching info --
    n_h0,n_w0 = search.query_batch_info(vid.shape) # just showing api
    ntotal = t * n_h0 * n_w0
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1
    qindex = 0

    # -- ground-truth --
    dists_gt,inds_gt = search(vid,qindex,ntotal,vid1=vidr)
    if k == -1:
        dists_gt = rearrange(dists_gt,'(sh sw) (h w) -> h w sh sw',sh=n_h0,h=n_h1)

    # -- testing code --
    dists_te = l2_dists(vid,inds_gt,qindex,vid1=vidr)
    if k == -1:
        dists_te = rearrange(dists_te,'(sh sw) (h w) -> h w sh sw',sh=n_h0,h=n_h1)

    # -- compare --
    tol = 1e-5
    error = th.mean(th.abs(dists_te - dists_gt)/dists_gt.abs()).item()
    if error > tol: print("error: ",error)
    assert error < tol

    tol = 1e-4
    max_error = th.abs((dists_te - dists_gt)/dists_gt.abs()).max().item()
    if max_error > tol: print("max error: ",max_error)
    assert max_error < tol

def test_cu_vs_th_bwd(k,ps,stride0,stride1,dilation,reflect_bounds,exact):
    """

    Test the CUDA code with torch code

    Forward Pass

    """
    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    k,pt = 1,1
    wt = 0
    ws = -1
    # stride0 = stride
    # stride1 = 1
    search_abs = True
    use_k = k>0
    exact = True

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    adj = True
    only_full = True

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device)[:1,].contiguous()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],-2)

    # -- normalize --
    vid /= vid.max()
    # vidr = th.ones_like(vid)
    vidr = th.rand_like(vid)

    # -- allow for grads --
    vid0_te = vid.clone()
    vid1_te = vidr.clone()
    vid0_gt = vid.clone()
    vid1_gt = vidr.clone()
    vid0_te.requires_grad_(True)
    vid1_te.requires_grad_(True)
    vid0_gt.requires_grad_(True)
    vid1_gt.requires_grad_(True)

    # -- compute flow --
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape
    chnls = vid.shape[1]

    # -- unpack --
    _,_,n0,n1 = get_batching_info(vid.shape,stride0,stride1,ps,dil)
    n_h0,n_w0 = n0[0],n0[1]
    n_h1,n_w1 = n1[0],n1[1]

    # -- exec fold fxns --
    use_adj = True
    h0_off,w0_off,_,_ = comp_pads(vid.shape, ps, stride0, 1)
    h1_off,w1_off,_,_ = comp_pads(vid.shape, ps, stride1, 1)
    search = dnls.search.init("l2_with_index",
                              flows.fflow, flows.bflow, k, ps, pt,
                              ws, wt, dilation=dil,
                              stride0=stride0, stride1=stride1,
                              use_k = use_k,use_adj=use_adj,
                              reflect_bounds=reflect_bounds,
                              search_abs=search_abs,exact=exact,
                              h0_off=h0_off,w0_off=w0_off,
                              h1_off=h1_off,w1_off=w1_off)
    l2_dists = dnls.search.init("l2_dists", ps, pt,
                                dilation=dil,stride0=stride0,
                                use_adj=use_adj,
                                reflect_bounds=reflect_bounds,
                                exact=exact,
                                h0_off=h0_off,w0_off=w0_off,
                                h1_off=h1_off,w1_off=w1_off)

    # -- batching info --
    n_h0,n_w0 = search.query_batch_info(vid.shape) # just showing api
    ntotal = t * n_h0 * n_w0
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1
    qindex = 0

    # -- ground-truth --
    dists_gt,inds_gt = search(vid0_gt,qindex,ntotal,vid1=vid1_gt)
    if k == -1:
        dists_gt = rearrange(dists_gt,'(sh sw) (h w) -> h w sh sw',sh=n_h0,h=n_h1)

    # -- testing code --
    dists_te = l2_dists(vid0_te,inds_gt,qindex,vid1=vid1_te)
    if k == -1:
        dists_te = rearrange(dists_te,'(sh sw) (h w) -> h w sh sw',sh=n_h0,h=n_h1)

    # -- backward --
    dists_grad = th.rand_like(dists_gt)
    th.autograd.backward(dists_gt,dists_grad)
    th.autograd.backward(dists_te,dists_grad)

    # -- unpack grads --
    grad0_te = vid0_te.grad
    grad1_te = vid1_te.grad
    grad0_gt = vid0_gt.grad
    grad1_gt = vid1_gt.grad


    #
    # -- Backward Step --
    #

    # -- tolerances --
    small_thresh = 1e-2
    tol_mean = 1e-8
    tol_max = 5e-10

    # -- check 0 --
    args = th.where(grad0_gt.abs() > small_thresh)
    diff = th.abs((grad0_te - grad0_gt)/(grad0_gt.abs()+1e-5))
    error = diff.mean().item()
    assert error < tol_mean
    error = diff[args].max().item()
    assert error < tol_max

    # -- check 1 --
    args = th.where(grad1_gt.abs() > small_thresh)
    diff = th.abs((grad1_te - grad1_gt)/(grad1_gt.abs()+1e-10))
    error = diff.mean().item()
    assert error < tol_mean
    error = diff[args].max().item()
    assert error < tol_max
