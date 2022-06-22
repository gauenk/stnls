
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

def print_gpu_stats(gpu_bool,note=""):
    if gpu_bool:
        gpu_max = th.cuda.memory_allocated()/(1024**3)
        print("[%s] GPU Max: %2.4f" % (note,gpu_max))

def pytest_generate_tests(metafunc):
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)
    # test_lists = {"ps":[3],"stride":[1],"dilation":[1],
    #               "k":[10],"ws":[10],"wt":[5],"pt":[1],"chnls":[1]}
    test_lists = {"ps":[3,5],"stride":[1,2,3,4,5],"dilation":[1,2,3,4,5],
                  "k":[1,10],"ws":[10],"wt":[5],"pt":[1],"chnls":[1]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

# -----------------------------------------
#
#   Testing Efficient vs. Simple Gather
#
# -----------------------------------------

@pytest.mark.skip()
def test_compare_efficient(k,ps,stride,dilation,ws,wt,pt,chnls):

    # -- set device --
    device = "cuda:0"
    th.cuda.set_device(device)
    sigma = 50.
    # dname = "text_tourbus_64"
    dname = "davis_baseball_64x64"

    # -- init vars --
    sigma = 30.
    clean_flow = True
    comp_flow = False
    if k > 1: exact = True
    else: exact = False

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext="jpg")
    vid = th.from_numpy(vid).to(device)
    noisy = vid + sigma * th.randn_like(vid)
    flow = dnls.testing.flow.get_flow(comp_flow,clean_flow,noisy,vid,sigma)

    # -- gather/scatter decl --
    scatter_nl = dnls.scatter.ScatterNl(ps,pt,dilation=dilation,
                                        exact=True,device=device)
    gather_simp = dnls.gather.GatherNl(vid.shape,dilation=dilation,
                                     exact=exact,device=device)
    gather_eff = dnls.gather.GatherNl(vid.shape,dilation=dilation,
                                      use_race=False,device=device)

    # -- batching info --
    device = noisy.device
    shape = noisy.shape
    t,c,h,w = shape
    nh = (h-1)//stride+1
    nw = (w-1)//stride+1
    ntotal = t * nh * nw
    qSize = 64
    nbatches = (ntotal-1) // qSize + 1
    vid = vid.contiguous()
    th.cuda.synchronize()

    # -- nbatches --
    for index in range(nbatches):

        # -- get [patches & nlInds] --
        queryInds = dnls.utils.inds.get_query_batch(index,qSize,stride,
                                                    t,h,w,device)
        nlDists,nlInds = dnls.simple.search.run(vid,queryInds,
                                                flow,k,ps,pt,ws,wt,chnls)
        patches = scatter_nl(vid,nlInds)

        # -- reset gather --
        gather_nl.vid[...] = 0
        gather_nl.wvid[...] = 0

        # -- testing forward --
        vid_simp,wvid_simp = gather_simp(patches,nlDists,nlInds)
        vid_eff,wvid_eff = gather_eff(patches,nlDists,nlInds)
        error = th.mean((vid_eff - vid_simp)**2).item()
        assert error < 1e-10

        # -- save --
        # vid_nl /= vid_nl.max()
        # vid_simp /= vid_simp.max()
        # dnls.testing.data.save_burst(vid_nl[[0]],SAVE_DIR,"nl_%d" % index)
        # dnls.testing.data.save_burst(vid_simp[[0]],SAVE_DIR,"simp_%d" % index)

    th.cuda.synchronize()

# --------------------------------------
#
#   Testing Gather vs. Simple Code
#
# -------------------------------------

def test_compare_simple(k,ps,stride,dilation,ws,wt,pt,chnls):

    # -- set device --
    device = "cuda:0"
    th.cuda.set_device(device)
    sigma = 50.
    # dname = "text_tourbus_64"
    dname = "davis_baseball_64x64"

    # -- init vars --
    sigma = 30.
    clean_flow = True
    comp_flow = False
    if k > 1: exact = True
    else: exact = False

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext="jpg")
    vid = th.from_numpy(vid).to(device)
    noisy = vid + sigma * th.randn_like(vid)
    flow = dnls.testing.flow.get_flow(comp_flow,clean_flow,noisy,vid,sigma)

    # -- gather/scatter decl --
    scatter_nl = dnls.scatter.ScatterNl(ps,pt,dilation=dilation,
                                        exact=True,device=device)
    gather_nl = dnls.gather.GatherNl(vid.shape,dilation=dilation,
                                     exact=exact,device=device)

    # -- batching info --
    device = noisy.device
    shape = noisy.shape
    t,c,h,w = shape
    nh = (h-1)//stride+1
    nw = (w-1)//stride+1
    ntotal = t * nh * nw
    qSize = 64
    nbatches = (ntotal-1) // qSize + 1
    vid = vid.contiguous()
    th.cuda.synchronize()

    # -- nbatches --
    for index in range(nbatches):

        # -- get [patches & nlInds] --
        queryInds = dnls.utils.inds.get_query_batch(index,qSize,stride,
                                                    t,h,w,device)
        nlDists,nlInds = dnls.simple.search.run(vid,queryInds,
                                                flow,k,ps,pt,ws,wt,chnls)
        patches = scatter_nl(vid,nlInds)

        # -- reset gather --
        gather_nl.vid[...] = 0
        gather_nl.wvid[...] = 0

        # -- testing forward --
        vid_nl,wvid_nl = gather_nl(patches,nlDists,nlInds)
        vid_simp,wvid_simp = dnls.simple.gather.run(patches,nlDists,
                                                    nlInds,dilation=dilation,
                                                    shape=shape)
        error = th.mean((vid_nl - vid_simp)**2).item()
        assert error < 1e-10

        # -- save --
        # vid_nl /= vid_nl.max()
        # vid_simp /= vid_simp.max()
        # dnls.testing.data.save_burst(vid_nl[[0]],SAVE_DIR,"nl_%d" % index)
        # dnls.testing.data.save_burst(vid_simp[[0]],SAVE_DIR,"simp_%d" % index)

    th.cuda.synchronize()

#
# -- Test Gather & Fold --
#

# @pytest.mark.skip()
def test_compare_fold(ps,stride,dilation,ws,wt,pt,chnls):

    # -- set device --
    device = "cuda:0"
    th.cuda.set_device(device)
    sigma = 50.
    # dname = "text_tourbus_64"
    dname = "davis_baseball_64x64"

    # -- init vars --
    k = 1
    clean_flow = True
    comp_flow = False
    exact = True

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext="jpg")
    vid = th.from_numpy(vid).to(device)
    noisy = vid + sigma * th.randn_like(vid)
    flow = dnls.testing.flow.get_flow(comp_flow,clean_flow,noisy,vid,sigma)

    # -- batching info --
    shape = noisy.shape
    t,c,h,w = shape
    nh = (h-1)//stride+1
    nw = (w-1)//stride+1
    ntotal = t * nh * nw
    qSize = ntotal
    nbatches = (ntotal-1) // qSize + 1
    vid = vid.contiguous()

    # -- exec gather fxns --
    scatter_nl = dnls.scatter.ScatterNl(ps,pt,dilation=dilation,
                                        exact=True,device=device)
    gather_nl = dnls.gather.GatherNl((t,c,h,w),dilation=dilation,
                                     exact=exact,device=device)

    # -- get [patches & nlInds] --
    index = 0
    queryInds = dnls.utils.inds.get_query_batch(index,qSize,stride,
                                                t,h,w,device)
    nlDists,nlInds = dnls.simple.search.run(vid,queryInds,
                                            flow,k,ps,pt,ws,wt,chnls)
    patches = scatter_nl(vid,nlInds)

    #
    # -- test logic --
    #

    # -- prepare videos --
    patches_nn = patches
    patches_nl = patches.clone()
    patches_nn.requires_grad_(True)
    patches_nl.requires_grad_(True)

    # -- run forward --
    vid_nn,wvid_nn = run_fold(patches_nn,t,h,w,stride,dilation)
    vid_nl,wvid_nl = gather_nl(patches_nl,nlDists,nlInds)

    # -- reweight --
    args = th.where(wvid_nl>0)
    vid_nl[args] /= wvid_nl[args]
    vid_nn[args] /= wvid_nn[args]

    # -- run backward --
    vid_grad = th.randn_like(vid)
    th.autograd.backward(vid_nn,vid_grad)
    th.autograd.backward(vid_nl,vid_grad)

    # -- check forward --
    diff = th.abs(vid_nn - vid_nl)
    error = th.mean((vid_nn - vid_nl)**2).item()
    assert error < 1e-10

    # -- check backward --
    grad_nn = patches_nn.grad
    grad_nl = patches_nl.grad
    if exact: tol = 1e-10
    else: tol = 1.
    error = th.mean((grad_nn - grad_nl)**2).item()
    assert error < tol

#
# -- Helpers --
#


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

def run_unfold(vid,ps):
    psHalf = ps//2
    shape_str = 't (c h w) np -> (t np) 1 1 c h w'
    vid_pad = pad(vid,4*[psHalf,],mode="reflect")
    patches = unfold(vid_pad,(ps,ps))
    patches = rearrange(patches,shape_str,h=ps,w=ps)
    return patches

