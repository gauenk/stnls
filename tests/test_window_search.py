"""
Write "inds" check;
verify the correct meshgrid is included in the exhaustive search

see pacnet's test for details.


"""

# -- python --
import cv2,tqdm,copy,pytest
import numpy as np
import unittest
import tempfile
import sys
import shutil
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

# -- misc --
import torch.nn.functional as nnf

# -- check if reordered --
from scipy import optimize
SAVE_DIR = Path("./output/tests/window_search")

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
    test_lists = {"ps":[1],"stride0":[1],"stride1":[1],"dilation":[1],"wt":[0],
                  "ws":[-1],"k":[-1],"exact":[True],"reflect_bounds":[False],
                  "nheads":[2]}
    # test_lists = {"ps":[3,4,5,6,7,8],"stride":[1,2,3,4,5,8],"dilation":[1,2,3,4,5,8],
    #               "top":[1,11],"btm":[50,57],"left":[3,7],"right":[57,30]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

#
# -- forward testing --
#

def test_cu_vs_th_fwd(ps,stride0,stride1,nheads,dilation,reflect_bounds,exact):
    """

    Test the CUDA code with torch code

    Forward Pass

    """

    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    pt = 1
    wt = 0
    ws = 8
    k = -1
    # stride0 = stride
    # stride1 = 1
    search_abs = False
    use_k = k>0
    exact = True
    use_adj = True

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    adj = False
    only_full = False

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device)[:1,].contiguous()
    vid = repeat(vid,'t c h w -> t (r c) h w',r=12)[:,:16] # want 32 channels
    vid = th.cat([vid,vid],1)
    vid = th.cat([vid,vid],-1)
    vid = th.cat([vid,vid],-2)
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

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

    # -- init --
    adj = 0
    search = dnls.search.init("window",
                              flows.fflow, flows.bflow, k, ps, pt,
                              ws, wt, nheads, dilation=dil,
                              stride0=stride0, stride1=stride1,
                              use_k = use_k,use_adj=use_adj,
                              reflect_bounds=reflect_bounds,
                              search_abs=search_abs,exact=exact,
                              h0_off=0,w0_off=0,h1_off=0,w1_off=0)
    wpsum = dnls.reducers.WeightedPatchSumHeads(ps,pt,dilation=dil,adj=adj,
                                                reflect_bounds=reflect_bounds,
                                                exact=exact)
    fold = dnls.iFoldz(vid.shape,None,stride=stride0,dilation=dil,
                       adj=adj,only_full=only_full,
                       use_reflect=reflect_bounds,device=device)


    # -- batching info --
    n_h0,n_w0 = search.query_batch_info(vid.shape) # just showing api
    ntotal = t * n_h0 * n_w0
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    #
    #
    # -- Primary Code Comparison --
    #
    #

    # -- ground-truth --
    vid_gt,dists_gt = dnls.simple.window_search.run(vid,ws,nheads)
    dists_gt = rearrange(dists_gt,'H (b d1) d2 -> H b d1 d2',d1=ws*ws)

    # -- vid ours --
    dists_te,inds_te = search(vid,0,ntotal)
    # dists_s = dists_te
    dists_s = search.window_softmax(dists_te,vid.shape)
    patches = wpsum(vid,dists_s,inds_te)

    # -- folding [v1] --
    # patches = rearrange(patches,'b H c 1 1 -> b (H c)')
    # vid_te = rearrange(patches,'(h w) c -> 1 c h w',h=128)

    # -- folding [v2] --
    patches = rearrange(patches,'b H c 1 1 -> b 1 1 (H c) 1 1')
    fold(patches,0)
    vid_te = fold.vid / fold.zvid

    # -- shape to match --
    dists_te,_ = search.match_simple(dists_te,inds_te,vid.shape)

    # -- viz --
    vid_gt_s = vid_gt / vid_gt.max()
    dnls.testing.data.save_burst(vid_gt_s[:,-3:],SAVE_DIR,'vid_gt')
    vid_te_s = vid_te / vid_te.max()
    dnls.testing.data.save_burst(vid_te_s[:,-3:],SAVE_DIR,'vid_te')

    # diff = th.abs(vid_gt - vid_te)
    # args = th.where(diff > 1e-3)

    # diff = th.abs(vid_gt - vid_te).mean(1,keepdim=True)
    # diff_s = diff / diff.max()
    # dnls.testing.data.save_burst(diff_s,SAVE_DIR,'vid_diff')

    # -- testing [dists] --
    diff = th.abs(dists_gt - dists_te)/(dists_gt.abs()+1e-5)

    error = diff.mean().item()
    assert error < 1e-5

    error = diff.max().item()
    assert error < 1e-4

    # -- testing [vid] --
    diff = th.abs(vid_gt - vid_te)/(vid_gt.abs()+1e-5)

    error = diff.mean().item()
    assert error < 1e-5

    error = diff.max().item()
    assert error < 1e-4


def test_cu_vs_th_bwd(ps,stride0,stride1,nheads,dilation,reflect_bounds,exact):
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
    k = -1
    # stride0 = stride
    # stride1 = 1
    search_abs = False
    use_k = k>0
    exact = True
    use_adj = True

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
    vid = repeat(vid,'t c h w -> t (r c) h w',r=12)[:,:32] # want 32 channels
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

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

    # -- init --
    adj = 0
    search = dnls.search.init("window",
                              flows.fflow, flows.bflow, k, ps, pt,
                              ws, wt, nheads, dilation=dil,
                              stride0=stride0, stride1=stride1,
                              use_k = use_k,use_adj=use_adj,
                              reflect_bounds=reflect_bounds,
                              search_abs=search_abs,exact=exact,
                              h0_off=0,w0_off=0,h1_off=0,w1_off=0)
    wpsum = dnls.reducers.WeightedPatchSum(ps,pt,dilation=dil,adj=adj,
                                           reflect_bounds=reflect_bounds,
                                           nbwd=1,exact=exact)
    fold = dnls.iFoldz(vid.shape,None,stride=stride0,dilation=dil,
                       adj=adj,only_full=only_full,
                       use_reflect=reflect_bounds,device=device)


    # -- batching info --
    n_h0,n_w0 = search.query_batch_info(vid.shape) # just showing api
    ntotal = t * n_h0 * n_w0
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    #
    #
    # -- Primary Code Comparison --
    #
    #

    # -- allow grads --
    in_vid_gt = vid.clone()
    in_vid_te = vid.clone()
    in_vid_gt.requires_grad_(True)
    in_vid_te.requires_grad_(True)

    # -- ground-truth --
    vid_gt = dnls.simple.window_search.run(in_vid_gt,8)

    # -- vid ours --
    dists,inds = search(in_vid_te,0,ntotal)
    patches = wpsum(in_vid_te,dists,inds)
    patches = rearrange(patches,'b H c h w -> b 1 1 (H c) h w')
    fold(patches,0)
    vid_te = fold.vid / fold.zvid

    # -- viz --
    vid_gt_s = vid_gt / vid_gt.max()
    dnls.testing.data.save_burst(vid_gt_s[:,:3],SAVE_DIR,'vid_gt')
    vid_te_s = vid_te / vid_te.max()
    dnls.testing.data.save_burst(vid_te_s[:,:3],SAVE_DIR,'vid_te')

    # -- backward --
    vid_grad = th.randn_like(vid_gt)
    th.autograd.backward(vid_gt,vid_grad)
    th.autograd.backward(vid_te,vid_grad)
    grad_gt = vid_gt.grad
    grad_te = vid_te.grad

    # -- testing --
    diff = th.abs(grad_gt - grad_te)/(grad_gt.abs()+1e-5)

    error = diff.mean().item()
    assert error < 1e-5

    error = diff.max().item()
    assert error < 1e-4
