
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
SAVE_DIR = Path("./output/tests/prod_search_with_heads")

def pytest_generate_tests(metafunc):
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)
    test_lists = {"ps":[7],"stride0":[4],"stride1":[4],
                  "dilation":[1],"wt":[0],"ws":[-1],
                  "k":[-1],"exact":[True],"nheads":[1],
                  "seed":[0]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def test_fwd(ws,wt,k,ps,stride0,stride1,dilation,nheads,exact,seed):
    """

    Test the CUDA code with torch code

    Forward Pass

    """


    # -- get args --
    dil = dilation
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    pt = 1
    set_seed(seed)

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    reflect_bounds = True
    use_k = k > 0
    use_adj = False
    adj = 0
    full_ws = True

    # -- load data --
    vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid.to(device)[:,:5,].contiguous()
    vid = repeat(vid,'b t c h w -> b t (r c) h w',r=12)[:,:32].contiguous()
    vid /= vid.max()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)
    flows.fflow = 10*th.randn_like(flows.fflow)
    flows.bflow = 10*th.randn_like(flows.bflow)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape
    chnls = vid.shape[2]

    # -- pads --
    _,_,n0,n1 = get_batching_info(vid[0].shape,stride0,stride1,ps,dil)
    n_h0,n_w0 = n0[0],n0[1]
    n_h1,n_w1 = n1[0],n1[1]
    h0_off, w0_off, h1_off, w1_off = 0, 0, 0, 0

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h0 * n_w0
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- exec fold fxns --
    search_te = stnls.search.init("prod_patches_with_heads",flows.fflow, flows.bflow,
                                 k, ps, pt, ws, wt, nheads,
                                 chnls=-1,dilation=dil,
                                 stride0=stride0, stride1=stride1,
                                 reflect_bounds=reflect_bounds,use_k=use_k,
                                 h0_off=h0_off, w0_off=w0_off,
                                 h1_off=h1_off, w1_off=w1_off,
                                 search_abs=False,use_adj=use_adj,
                                 full_ws=full_ws,exact=exact)
    search_gt = stnls.search.init("prod_with_heads",flows.fflow, flows.bflow,
                                 k, ps, pt, ws, wt,
                                 h0_off, w0_off, h1_off, w1_off,
                                 chnls=-1,dilation=dil,
                                 stride0=stride0, stride1=stride1,
                                 reflect_bounds=reflect_bounds,use_k=use_k,
                                 search_abs=False,use_adj=use_adj,
                                 full_ws=full_ws,exact=exact)

    # -- [testing] search --
    patches = search_te.unfold(vid)
    dists_te,inds_te = search_te(patches,0,ntotal)

    # -- [groundtruth] search --
    dists_gt,inds_gt = search_gt(vid,0,ntotal)
    th.cuda.synchronize()

    # -- viz --
    # print(dists_te[0,0,0,:])
    # print(dists_gt[0,0,0,:])
    # print(dists_te[0,0,1,:10])
    # print(dists_gt[0,0,1,:10])
    # print(dists_te.shape)
    # print(dists_gt.shape)

    # -- viz --
    # diff = th.abs(dists_te - dists_gt).mean((-1,-2))
    # if diff.max() > 1e-5: diff /= diff.max()
    # diff = repeat(diff,'h w -> 1 c h w',c=3)
    # stnls.testing.data.save_burst(diff,SAVE_DIR,"nn2_diff")

    # diff = th.abs(dists_te - dists_gt).mean((0,1))
    # if diff.max() > 1e-5: diff /= diff.max()
    # diff = repeat(diff,'h w -> 1 c h w',c=3)
    # stnls.testing.data.save_burst(diff,SAVE_DIR,"nn2_diff_t")

    # -- compare --
    args0 = th.where(th.logical_not(th.isinf(dists_gt))) # remove all inf
    diff = th.abs(dists_te - dists_gt) / (dists_gt.abs()+1e-5)
    diff = diff[args0]

    # -- test --
    tol = 1e-5
    error = diff.mean().item()
    if error > tol: print("error: ",error)
    assert error < tol

    tol = 1e-4
    max_error = diff.max().item()
    if max_error > tol: print("max error: ",max_error)
    assert max_error < tol

@pytest.mark.slow
def test_bwd(ws,wt,k,ps,stride0,stride1,dilation,nheads,exact,seed):
    """

    Test the CUDA code with torch code

    Forward Pass

    """


    # -- get args --
    dil = dilation
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    pt = 1

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    reflect_bounds = True
    search_abs = ws == -1
    use_k = k > 0
    use_adj = False
    full_ws = True
    adj = 0

    # -- load data --
    vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid.to(device)[:,:2,].contiguous()
    vid = repeat(vid,'b t c h w -> b t (r c) h w',r=12)[:,:32].contiguous()
    vid = vid[...,:32,:32]
    vid /= vid.max()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)
    flows.fflow = 10*th.randn_like(flows.fflow)
    flows.bflow = 10*th.randn_like(flows.bflow)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape
    chnls = vid.shape[2]

    # -- pads --
    _,_,n0,n1 = get_batching_info(vid[0].shape,stride0,stride1,ps,dil)
    n_h0,n_w0 = n0[0],n0[1]
    n_h1,n_w1 = n1[0],n1[1]
    h0_off, w0_off, h1_off, w1_off = 0, 0, 0, 0

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h0 * n_w0
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- allow grads --
    vid_te0,vid_te1 = vid.clone(),vid.clone()
    vid_te0.requires_grad_(True)
    vid_te1.requires_grad_(True)
    vid_gt0,vid_gt1 = vid.clone(),vid.clone()
    vid_gt0.requires_grad_(True)
    vid_gt1.requires_grad_(True)

    # -- exec fold fxns --
    search_te = stnls.search.init("prod_patches_with_heads",flows.fflow, flows.bflow,
                                 k, ps, pt, ws, wt, nheads,
                                 chnls=-1,dilation=dil,
                                 stride0=stride0, stride1=stride1,
                                 reflect_bounds=reflect_bounds,use_k=use_k,
                                 h0_off=h0_off, w0_off=w0_off,
                                 h1_off=h1_off, w1_off=w1_off,
                                 search_abs=search_abs,use_adj=use_adj,
                                 full_ws=full_ws,exact=exact)
    search_gt = stnls.search.init("prod_with_heads",flows.fflow, flows.bflow,
                                 k, ps, pt, ws, wt,
                                 h0_off, w0_off, h1_off, w1_off,
                                 chnls=-1,dilation=dil,
                                 stride0=stride0, stride1=stride1,
                                 reflect_bounds=reflect_bounds,use_k=use_k,
                                 search_abs=search_abs,use_adj=use_adj,
                                 full_ws=full_ws,exact=exact)


    # -- [testing] search --
    patches0 = search_te.unfold(vid_te0)
    patches1 = search_te.unfold(vid_te1)
    dists_te,inds_te = search_te(patches0,0,ntotal,patches1)

    # -- [groundtruth] search --
    dists_gt,inds_gt = search_gt(vid_gt0,0,ntotal,vid_gt1)
    th.cuda.synchronize()

    # -- viz --
    # print(dists_te[0,0,:10])
    # print(dists_gt[0,0,:10])
    # print(dists_te.shape)
    # print(dists_gt.shape)

    # -- viz --
    # diff = th.abs(dists_te - dists_gt).mean((-1,-2))
    # if diff.max() > 1e-5: diff /= diff.max()
    # diff = repeat(diff,'h w -> 1 c h w',c=3)
    # stnls.testing.data.save_burst(diff,SAVE_DIR,"nn2_diff")

    # diff = th.abs(dists_te - dists_gt).mean((0,1))
    # if diff.max() > 1e-5: diff /= diff.max()
    # diff = repeat(diff,'h w -> 1 c h w',c=3)
    # stnls.testing.data.save_burst(diff,SAVE_DIR,"nn2_diff_t")

    # -- compare --
    diff = th.abs(dists_te - dists_gt) / (dists_gt.abs()+1e-5)

    tol = 1e-5
    error = diff.mean().item()
    if error > tol: print("error: ",error)
    assert error < tol

    tol = 1e-4
    max_error = diff.max().item()
    if max_error > tol: print("max error: ",max_error)
    assert max_error < tol

    # -- compute bwd --
    dists_grad = th.randn_like(dists_te)
    th.autograd.backward(dists_te,dists_grad)
    th.autograd.backward(dists_gt,dists_grad)

    # -- for both grads --
    _grads_te = [vid_te0.grad,vid_te1.grad]
    _grads_gt = [vid_gt0.grad,vid_gt1.grad]
    for idx,(grads_te,grads_gt) in enumerate(zip(_grads_te,_grads_gt)):

        # -- viz [the error map may look weird] --
        # print("-"*20)
        # print(grads_te[0,-1,-3:,-3:])
        # print(grads_gt[0,-1,-3:,-3:])
        # print("-"*20)
        # print(grads_te[0,0,-3:,-3:])
        # print(grads_gt[0,0,-3:,-3:])
        # print("-"*20)
        # print(grads_te[0,0,10:13,10:13])
        # print(grads_gt[0,0,10:13,10:13])
        # print("-"*20)
        # print(grads_te[0,0,:3,:3])
        # print(grads_gt[0,0,:3,:3])
        # print("-"*20)

        # diff = (grads_te -grads_gt).abs()/(grads_gt.abs()+1e-8)
        # print(diff.max())
        # diff /= diff.max()
        # stnls.testing.data.save_burst(diff[:,[0]],SAVE_DIR,"grad_diff_0_%d" % exact)
        # stnls.testing.data.save_burst(diff[:,[1]],SAVE_DIR,"grad_diff_1_%d" % exact)
        # stnls.testing.data.save_burst(diff[:,[2]],SAVE_DIR,"grad_diff_2_%d" % exact)

        # -- compare grads --
        rel_error = th.abs(grads_gt - grads_te)/(th.abs(grads_gt)+1e-10)
        rel_error_nz  = rel_error

        tol = 1e-3
        error = th.max(rel_error_nz).item()
        if error > tol: print("Max Error: ",error)
        # print("Max Error: ",error)
        assert error < tol

        tol = 1e-4
        error = th.mean(rel_error_nz).item()
        if error > tol: print("Mean Error: ",error)
        # print("Mean Error: ",error)
        assert error < tol

