
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
from dnls.utils.inds import get_batching_info

# -- meshgrid --
import cache_io

# -- test func --
from torch.nn.functional import fold,unfold,pad
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/prod_search")

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
    test_lists = {"ps":[7],"stride0":[2,4],"stride1":[2,4],
                  "dilation":[1],"wt":[0,1,2,3],"ws":[15,23,32],
                  "k":[5],"exact":[True],"nheads":[1],
                  "seed":[0,1,2,3,4]}
    # test_lists = {"ps":[3,4,5,6,7,8],"stride":[1,2,3,4,5,8],"dilation":[1,2,3,4,5,8],
    #               "top":[1,11],"btm":[50,57],"left":[3,7],"right":[57,30]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)

def test_cu_vs_th_fwd(ws,wt,k,ps,stride0,stride1,dilation,nheads,exact,seed):
    """

    Test the CUDA code with torch code

    Forward Pass

    """


    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    pt = 1

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    reflect_bounds = True
    use_k = k > 0
    use_adj = False
    adj = 0

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device)[:5,].contiguous()
    vid = repeat(vid,'t c h w -> t (r c) h w',r=12)[:,:32].contiguous()
    vid /= vid.max()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- compute flow --
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)
    flows.fflow = 10*th.randn_like(flows.fflow)
    flows.bflow = 10*th.randn_like(flows.bflow)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape
    chnls = vid.shape[1]

    # -- pads --
    _,_,n0,n1 = get_batching_info(vid.shape,stride0,stride1,ps,dil)
    n_h0,n_w0 = n0[0],n0[1]
    n_h1,n_w1 = n1[0],n1[1]
    h0_off, w0_off, h1_off, w1_off = 0, 0, 0, 0

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h0 * n_w0
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- exec fold fxns --
    search_te = dnls.search.init("prod_with_heads",flows.fflow, flows.bflow,
                                 k, ps, pt, ws, wt, nheads,
                                 chnls=-1,dilation=dil,
                                 stride0=stride0, stride1=stride1,
                                 reflect_bounds=reflect_bounds,use_k=use_k,
                                 h0_off=h0_off, w0_off=w0_off,
                                 h1_off=h1_off, w1_off=w1_off,
                                 search_abs=False,use_adj=use_adj,
                                 exact=exact)
    search_gt = dnls.search.init("prod_with_index",flows.fflow, flows.bflow,
                                 k, ps, pt, ws, wt,
                                 h0_off, w0_off, h1_off, w1_off,
                                 chnls=-1,dilation=dil,
                                 stride0=stride0, stride1=stride1,
                                 reflect_bounds=reflect_bounds,use_k=use_k,
                                 use_search_abs=False,use_adj=use_adj,
                                 exact=exact)


    # -- [testing] search --
    # print("vid.shape: ",vid.shape)
    dists_te,inds_te = search_te(vid,0,ntotal)
    th.cuda.synchronize()

    # -- [groundtruth] search --
    search_te.nheads = 1
    _c = (vid.shape[1]-1) // nheads + 1
    dists_gt,inds_gt = [],[]
    for h in range(nheads):
        cinds = slice(_c*h,_c*(h+1))
        vid_c = vid[:,cinds]
        dists_h,inds_h = search_gt(vid_c,0,ntotal)
        # dists_h,inds_h = search_te(vid_c,0,ntotal)
        # dists_h,inds_h = dists_h[0],inds_h[0]
        dists_gt.append(dists_h)
        inds_gt.append(inds_h)
    dists_gt = th.stack(dists_gt)
    inds_gt  = th.stack(inds_gt)

    # -- viz --
    # print(dists_te[0,0,:10])
    # print(dists_gt[0,0,:10])
    # print(dists_te.shape)
    # print(dists_gt.shape)

    # -- viz --
    # diff = th.abs(dists_te - dists_gt).mean((-1,-2))
    # if diff.max() > 1e-5: diff /= diff.max()
    # diff = repeat(diff,'h w -> 1 c h w',c=3)
    # dnls.testing.data.save_burst(diff,SAVE_DIR,"nn2_diff")

    # diff = th.abs(dists_te - dists_gt).mean((0,1))
    # if diff.max() > 1e-5: diff /= diff.max()
    # diff = repeat(diff,'h w -> 1 c h w',c=3)
    # dnls.testing.data.save_burst(diff,SAVE_DIR,"nn2_diff_t")

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

