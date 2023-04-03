
# -- python --
import sys
import pytest

# -- data mgnmt --
from pathlib import Path
from easydict import EasyDict as edict

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- stnls --
import stnls

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
    test_lists = {"ps":[3],"stride":[1],"dilation":[1],
                  "k":[10],"ws":[7],"wt":[5],"pt":[1],"chnls":[1],
                  "clear_each":[True],}
    # test_lists = {"ps":[3,5],"stride":[1,2,3,4,5],"dilation":[1,2,3,4,5],
    #               "k":[1,10],"ws":[10],"wt":[5],"pt":[1],"chnls":[1]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

# -----------------------------------------
#
#   Testing Efficient vs. Simple FoldK
#
# -----------------------------------------

# @pytest.mark.skip()
def test_compare_efficient(k,ps,stride,dilation,ws,wt,pt,chnls,clear_each):

    # -- set device --
    device = "cuda:0"
    th.cuda.set_device(device)
    sigma = 50.
    # dname = "text_tourbus_64"
    # dname,ext = "davis_baseball_64x64","jpg"
    dname,ext = "text_bus","png"

    # -- init vars --
    sigma = 30.
    clean_flow = True
    comp_flow = False
    if k > 1: exact = True
    else: exact = False
    exact = True

    # -- load data --
    vid = stnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device)[:7]
    vid = th.cat([vid,vid],-1)
    vid = th.cat([vid,vid],-2)
    vid = th.cat([vid,vid],-2)
    vid = th.cat([vid,vid],-1)
    vid = vid.contiguous().clone()
    # print(vid.shape)
    noisy = vid + sigma * th.randn_like(vid)
    flow = stnls.flow.get_flow(comp_flow,clean_flow,noisy,vid,sigma)

    # -- unfold/fold k decl --
    unfold_k = stnls.UnfoldK(ps,pt,dilation=dilation,
                            exact=True,device=device)
    fold_k_simp = stnls.FoldK(vid.shape,dilation=dilation,
                             exact=exact,device=device)
    fold_k_eff = stnls.FoldK(vid.shape,dilation=dilation,
                            rand=True,nreps=1,device=device)

    # -- timer --
    timer = stnls.utils.timer.ExpTimer()

    # -- batching info --
    device = noisy.device
    shape = noisy.shape
    t,c,h,w = shape
    nh = (h-1)//stride+1
    nw = (w-1)//stride+1
    ntotal = t * nh * nw
    qSize = ntotal
    nbatches = (ntotal-1) // qSize + 1
    vid = vid.contiguous()
    th.cuda.synchronize()
    # print(ntotal)

    # -- nbatches --
    for index in range(nbatches):

        # -- get [patches & inds] --
        queryInds = stnls.utils.inds.get_query_batch(index,qSize,stride,
                                                    t,h,w,device)
        dists,inds = stnls.simple.search.run(vid,queryInds,
                                            flow,k,ps,pt,ws,wt,chnls)
        patches = unfold_k(vid,inds)
        patches[...] = 1.
        # print(inds[64*64-1])
        # print(inds[64*64])
        # print(inds[64*64+1])

        # -- find eq --
        eq = th.all(inds==0,-1)
        # print(eq.shape)
        args = th.where(eq)
        # print(args)
        # print(inds[...,0][args])
        # print(inds[...,1][args])
        # print(inds[...,2][args])

        # -- reset fold_k --
        if clear_each:
            fold_k_simp.vid[...] = 0
            fold_k_simp.wvid[...] = 0
            fold_k_eff.vid[...] = 0
            fold_k_eff.wvid[...] = 0

        # -- testing forward --
        timer.start("simp")
        vid_simp,wvid_simp = stnls.simple.fold_k.run(patches,dists,
                                                    inds,dilation=dilation,
                                                    shape=shape)
        # vid_simp,wvid_simp = fold_k_simp(patches,dists,inds)
        th.cuda.synchronize()
        timer.stop("simp")

        timer.start("eff")
        vid_eff,wvid_eff = fold_k_eff(patches,dists,inds)
        th.cuda.synchronize()
        timer.stop("eff")

        # -- viz --
        # print(timer)
        # print(vid_simp[0,0,:3,:3])
        # print(vid_eff[0,0,:3,:3])
        # print(vid_simp[0,0,16:19,16:19])
        # print(vid_eff[0,0,16:19,16:19])
        # print(vid_simp[6,0,16:19,16:19])
        # print(vid_eff[6,0,16:19,16:19])

        # -- exect most of the value to be mostly the same --
        diff = th.abs(vid_eff - vid_simp)/(th.abs(vid_simp)+1e-10)
        args0 = th.where(th.abs(vid_simp) > 1e-2) # rm small
        diff = diff[args0]
        args_big = th.where(diff >= 0.75) # get big % diff
        perc_big = len(args_big[0]) / float(diff.shape[0])
        assert perc_big < 1e-1 # one tenth is big
        args1 = th.where(diff < 0.75) # rm big % diff
        diff = diff[args1]
        ave_rel_error = th.mean(diff).item()
        assert ave_rel_error < 1e-1 # small on ave

        # -- save --
        # vid_nl /= vid_nl.max()
        # vid_simp /= vid_simp.max()
        # stnls.testing.data.save_burst(vid_nl[[0]],SAVE_DIR,"nl_%d" % index)
        # stnls.testing.data.save_burst(vid_simp[[0]],SAVE_DIR,"simp_%d" % index)

        # -- compare times --
        # print(timer)

    th.cuda.synchronize()

# --------------------------------------
#
#   Testing Fold_K vs. Simple Code
#
# -------------------------------------

# @pytest.mark.skip()
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
    vid = stnls.testing.data.load_burst("./data/",dname,ext="jpg")
    vid = th.from_numpy(vid).to(device)
    noisy = vid + sigma * th.randn_like(vid)
    flow = stnls.flow.get_flow(comp_flow,clean_flow,noisy,vid,sigma)

    # -- fold_k/unfold_k decl --
    unfold_k = stnls.UnfoldK(ps,pt,dilation=dilation,
                            exact=True,device=device)
    fold_k = stnls.FoldK(vid.shape,dilation=dilation,
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

        # -- get [patches & inds] --
        queryInds = stnls.utils.inds.get_query_batch(index,qSize,stride,
                                                    t,h,w,device)
        dists,inds = stnls.simple.search.run(vid,queryInds,
                                                flow,k,ps,pt,ws,wt,chnls)
        patches = unfold_k(vid,inds)

        # -- reset fold_k --
        fold_k.vid[...] = 0
        fold_k.wvid[...] = 0

        # -- testing forward --
        vid_nl,wvid_nl = fold_k(patches,dists,inds)
        vid_simp,wvid_simp = stnls.simple.fold_k.run(patches,dists,
                                                    inds,dilation=dilation,
                                                    shape=shape)
        error = th.mean((vid_nl - vid_simp)**2).item()
        assert error < 1e-10

        # -- save --
        # vid_nl /= vid_nl.max()
        # vid_simp /= vid_simp.max()
        # stnls.testing.data.save_burst(vid_nl[[0]],SAVE_DIR,"nl_%d" % index)
        # stnls.testing.data.save_burst(vid_simp[[0]],SAVE_DIR,"simp_%d" % index)

    th.cuda.synchronize()

#
# -- Test Fold_K & Fold --
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
    vid = stnls.testing.data.load_burst("./data/",dname,ext="jpg")
    vid = th.from_numpy(vid).to(device)
    noisy = vid + sigma * th.randn_like(vid)
    flow = stnls.flow.get_flow(comp_flow,clean_flow,noisy,vid,sigma)

    # -- batching info --
    shape = noisy.shape
    t,c,h,w = shape
    nh = (h-1)//stride+1
    nw = (w-1)//stride+1
    ntotal = t * nh * nw
    qSize = ntotal
    nbatches = (ntotal-1) // qSize + 1
    vid = vid.contiguous()

    # -- exec fold_k fxns --
    unfold_k = stnls.UnfoldK(ps,pt,dilation=dilation,
                            exact=True,device=device)
    fold_k = stnls.FoldK((t,c,h,w),dilation=dilation,
                                     exact=exact,device=device)

    # -- get [patches & inds] --
    index = 0
    queryInds = stnls.utils.inds.get_query_batch(index,qSize,stride,
                                                t,h,w,device)
    dists,inds = stnls.simple.search.run(vid,queryInds,
                                            flow,k,ps,pt,ws,wt,chnls)
    patches = unfold_k(vid,inds)
    dists = th.ones_like(dists)

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
    vid_nl,wvid_nl = fold_k(patches_nl,dists,inds)

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

