"""

Weighted-Patch (Inplace) Sum


Verbose Psuedo-Code:
   yi = softmax(dists)
   patches_i = unfold_k(b2,nlInds_cu).type(th.float64)
   patches_i = rearrange(patches_i,'n k 1 c h w -> n k (c h w)')
   zi = th.sum(yi * patches_i,1).type(th.float32) # n (c h w), this code!

"""


# -- python --
import sys

# -- data mgnmt --
from pathlib import Path
from easydict import EasyDict as edict

# -- testing --
import pytest

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- stnls --
import stnls
import stnls.utils.gpu_mem as gpu_mem
from stnls.utils.pads import same_padding,comp_pads


# -- meshgrid --


# -- test func --
from torch.nn.functional import fold,unfold,pad,softmax,log_softmax
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/reducer/wpsum/")

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)

def pytest_generate_tests(metafunc):
    seed = 123
    set_seed(seed)
    # test_lists = {"ps":[3],"stride":[1],"dilation":[1,2],
    test_lists = {"ps":[7],"stride":[4],"dilation":[1],"wt":[3],"k":[5],
                  "ws":[10],"top":[0],"btm":[-1],"left":[0],"right":[-1],
                  "exact":[True],"nheads":[1]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def test_forward(ps,stride,dilation,nheads,k,exact):

    # -- get args --
    dil = dilation
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    ext = "jpg"
    chnls,pt = 1,1
    stride0 = stride
    stride1 = 1
    ws = -1 if k == -1 else 10
    wt = 0 if k == -1 else 5

    # -- init vars --
    use_atomic = False
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    reflect_bounds = False
    use_k = k != -1
    use_unfold = k == -1
    t = 1 if use_unfold else 3
    # adj = ps//2 if use_unfold else 0
    use_adj = True

    # -- load data --
    vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)/255.
    vid = vid.to(device)

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape

    # -- init xsearch --
    dist_type = "l2"
    sch = stnls.search
    search = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                dist_type=dist_type,dilation=dil,
                                stride0=stride0, stride1=stride1,
                                reflect_bounds=reflect_bounds,anchor_self=True,
                                use_adj=use_adj)

    # -- init our inner product --
    wpsum = stnls.reducer.WeightedPatchSum(ps, pt, dilation=dil, use_adj=False,
                                           reflect_bounds=reflect_bounds,
                                           exact=exact, use_atomic=use_atomic)

    # -- run search & wpsum --
    scores,inds = search(vid,vid,flows.fflow,flows.bflow)
    scores_s = softmax(-scores*10,dim=-1)
    wpatches_te = wpsum(vid,scores_s,inds)#.view(scores_s.shape[0],-1)

    # -- ground-truth --
    wpatches_gt = stnls.simple.wpsum.run_patches(vid,scores_s,inds,ps,stride0,
                                                 use_adj=False,pt=pt,dilation=dilation,
                                                 reflect_bounds=reflect_bounds)

    # -- vis [scores] --
    # scores = scores.view(n_h,n_w,h,w)
    # scores = rearrange(scores,'nh nw h w -> h w nh nw')
    # scores_gt = scores_gt#.view(n_h,n_w,h,w)
    # print(scores[0,0,:3,:3])
    # print(scores_gt[0,0,:3,:3])

    # -- vis --
    # print(wpatches_te[:3,:3])
    # print(wpatches_gt[:3,:3])
    # diff = th.abs(wpatches_gt - wpatches_te).mean(-1)
    # diff = rearrange(diff,'(h w) -> 1 1 h w',h=32)
    # diff /= diff.max()
    # stnls.testing.data.save_burst(diff,SAVE_DIR,"diff_%d" % use_unfold)

    # -- compare --
    tol = 1e-5 if use_unfold else 1e-7
    error = th.abs(wpatches_gt - wpatches_te).mean().item()
    if error > tol: print(error)
    assert error < tol

    tol = 1e-4 if use_unfold else 1e-6
    error = th.abs(wpatches_gt - wpatches_te).max().item()
    if error > tol: print(error)
    assert error < tol


def test_score_backward(ps,stride,dilation,nheads,k):

    # -- get args --
    pt,dil = 1,dilation
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    ext = "jpg"
    stride0 = stride
    stride1 = 1
    ws = -1 if k == -1 else 10
    wt = 0 if k == -1 else 5

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    reflect_bounds = False
    search_abs = ws == -1
    use_k = k != -1
    use_unfold = k == -1
    t = 1 if use_unfold else 3
    use_adj = True
    exact = True
    use_atomic = True

    # -- load data --
    vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)/255.
    vid = vid.to(device)

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape

    # -- init xsearch --
    dist_type = "l2"
    sch = stnls.search
    search = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                dist_type=dist_type,dilation=dil,
                                stride0=stride0, stride1=stride1,
                                reflect_bounds=reflect_bounds,anchor_self=True,
                                use_adj=use_adj)

    # -- init our inner product --
    wpsum = stnls.reducer.WeightedPatchSum(ps, pt, dilation=dil, use_adj=False,
                                           reflect_bounds=reflect_bounds,
                                           exact=exact, use_atomic=use_atomic)

    # -- run search --
    scores,inds = search(vid,vid,flows.fflow,flows.bflow)
    scores_s = softmax(-scores*10,dim=-1)

    # -- forwards --
    scores_te = scores_s.clone()
    scores_te.requires_grad_(True)
    wpatches_te = wpsum(vid,scores_te,inds)#.view(scores_s.shape[0],-1)

    # -- ground-truth --
    scores_gt = scores_s.clone()
    scores_gt.requires_grad_(True)
    wpatches_gt = stnls.simple.wpsum.run_patches(vid,scores_gt,inds,ps,stride0,
                                                 use_adj=False,pt=pt,dilation=dilation,
                                                 reflect_bounds=reflect_bounds)

    # -- confirm fwd --
    tol = 1e-5 if use_unfold else 1e-7
    error = th.abs(wpatches_te - wpatches_gt).mean().item()
    if error > tol: print(error)
    assert error < tol

    # -- backward passes --
    wpatches_grad = th.rand_like(wpatches_te)
    th.autograd.backward(wpatches_te,wpatches_grad)
    th.autograd.backward(wpatches_gt,wpatches_grad)

    # -- set tol --
    tol_mean = 1e-5
    tol_max = 1e-3

    # -- grab grads --
    _grads_te = [scores_te.grad]
    _grads_gt = [scores_gt.grad]
    for idx,(grads_te,grads_gt) in enumerate(zip(_grads_te,_grads_gt)):

        # -- compute error --
        diff = th.abs((grads_te - grads_gt)/(grads_gt.abs()+1e-10))
        args = th.where(grads_gt.abs() > 1e-1)

        # -- viz --
        # print(len(args[0]),len(grads_gt.ravel()),grads_gt.abs().mean())
        # args2 = th.where(diff[args] > 0.003)
        # print(grads_gt[args][args2],grads_te[args][args2])

        # -- compare --
        error = diff.mean().item()
        if exact:
            if error > tol_mean: print("mean error: ",error)
            assert error < tol_mean

        error = diff[args].max().item()
        if exact:
            if error > tol_max: print("max error: ",error)
            assert error < tol_max

# @pytest.mark.slow
def test_vid_backward(ps,stride,dilation,nheads,k):

    # -- get args --
    pt,dil = 1,dilation
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    ext = "jpg"
    stride0 = stride
    stride1 = 1
    ws = -1 if k == -1 else 10
    wt = 0 if k == -1 else 5

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    reflect_bounds = False
    search_abs = ws == -1
    use_k = k != -1
    use_unfold = k == -1
    t = 1 if use_unfold else 3
    use_adj = True
    exact = True
    use_atomic = True

    # -- load data --
    vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)/255.
    vid = vid.to(device)

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape

    # -- init xsearch --
    dist_type = "l2"
    sch = stnls.search
    search = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                dist_type=dist_type,dilation=dil,
                                stride0=stride0, stride1=stride1,
                                reflect_bounds=reflect_bounds,anchor_self=True,
                                use_adj=use_adj)

    # -- init our inner product --
    wpsum = stnls.reducer.WeightedPatchSum(ps, pt, dilation=dil, use_adj=False,
                                           reflect_bounds=reflect_bounds,
                                           exact=False, use_atomic=use_atomic)

    # -- run search --
    scores,inds = search(vid,vid,flows.fflow,flows.bflow)
    # scores[:,:,1:] = 0
    scores_s = softmax(-scores*100,dim=-1)
    # scores_s[:,:,1:] = 0
    # print("scores_s.shape: ",scores_s.shape)
    # print(scores[0,0,0])
    # print(scores_s[0,0,0])
    # print(inds[0,0,0])

    # -- require grads  --
    vid_te = vid.clone()
    vid_te = vid_te.requires_grad_(True)
    vid_gt = vid.clone()
    vid_gt = vid_gt.requires_grad_(True)

    # -- forwards --
    wpatches_te = wpsum(vid_te,scores_s,inds)

    # -- ground-truth --
    wpatches_gt = stnls.simple.wpsum.run_patches(vid_gt,scores_s,inds,ps,stride0,
                                                 use_adj=False,pt=pt,dilation=dilation,
                                                 reflect_bounds=reflect_bounds)

    # -- confirm fwd --
    tol = 1e-5 if use_unfold else 1e-7
    error = th.abs(wpatches_te - wpatches_gt).mean().item()
    if error > tol: print(error)
    assert error < tol

    # -- backward passes --
    wpatches_grad = th.rand_like(wpatches_te)
    th.autograd.backward(wpatches_te,wpatches_grad)
    th.autograd.backward(wpatches_gt,wpatches_grad)

    # -- set tol --
    tol_mean = 1e-7
    tol_max = 1e-6

    # -- grab grads --
    _grads_te = [vid_te.grad]
    _grads_gt = [vid_gt.grad]
    for idx,(grads_te,grads_gt) in enumerate(zip(_grads_te,_grads_gt)):

        # -- viz --
        # print("-"*30)
        # print("grads_te.shape: ",grads_te.shape)
        # print("--- grads_te ---")
        # print(grads_te[0,:,0,:5,:5])
        # print("--- grads_gt ---")
        # print(grads_gt[0,:,0,:5,:5])
        # print("--- gt/te ---")
        # print(grads_gt[0,:,0,:5,:5]/grads_te[0,:,0,:5,:5])
        # print(grads_gt[0,:,1,:5,:5]/grads_te[0,:,1,:5,:5])
        # print("-"*30)

        # -- compute error --
        diff = th.abs((grads_te - grads_gt)/(grads_gt.abs()+1e-10))
        args = th.where(grads_gt.abs() > 1e-1)

        # -- viz --
        # print(len(args[0]),len(grads_gt.ravel()),grads_gt.abs().mean())
        # args2 = th.where(diff[args] > 0.003)
        # print(grads_gt[args][args2],grads_te[args][args2])

        # -- compare --
        error = diff.mean().item()
        if exact:
            if error > tol_mean: print("mean error: ",error)
            assert error < tol_mean

        error = diff[args].max().item()
        if exact:
            if error > tol_max: print("max error: ",error)
            assert error < tol_max
