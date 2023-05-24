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

def get_data(dnames,ext="jpg",device="cuda:0"):
    vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid.to(device)[:,:5,].contiguous()
    vid = repeat(vid,'b t c h w -> b t (r c) h w',r=12)[:,:32].contiguous()
    vid /= vid.max()
    return vid

def pytest_generate_tests(metafunc):
    seed = 123
    set_seed(seed)
    test_lists = {"ps":[7],"pt":[1],"ws":[10],"wt":[3],
                  "stride0":[4],"stride1":[1],
                  "dilation":[1],"k":[5],
                  "nheads":[1],"batchsize":[-1]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def test_forward(ps,pt,ws,wt,stride0,stride1,dilation,nheads,k,batchsize):

    # -- init vars --
    use_atomic = False
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    reflect_bounds = False
    use_adj = False

    # -- load data --
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    vid = get_data(dnames)

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)

    # -- init xsearch --
    dist_type = "l2"
    sch = stnls.search
    search = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                dist_type=dist_type,dilation=dilation,
                                stride0=stride0, stride1=stride1,
                                reflect_bounds=reflect_bounds,anchor_self=True,
                                use_adj=use_adj)

    # -- ground-truth --
    wpsum_gt = stnls.reducer.FoldedWeightedPatchSum(ps, stride0, -1,
                                                    pt, dilation=dilation,
                                                    use_adj=use_adj,
                                                    reflect_bounds=reflect_bounds,
                                                    use_atomic=use_atomic)


    # -- testing --
    wpsum_te = stnls.reducer.WeightedPatchSum(ps, pt, dilation=dilation,
                                           use_adj=use_adj,
                                           reflect_bounds=reflect_bounds,
                                           use_atomic=use_atomic)
    fold = stnls.iFoldz(vid.shape,stride=stride0,dilation=dilation,
                        use_adj=use_adj,reflect_bounds=reflect_bounds,
                        device=vid.device)


    # -- run search & wpsum --
    scores,inds = search(vid,vid,flows.fflow,flows.bflow)
    scores_s = softmax(-scores*10,dim=-1)

    # -- groundtruth --
    wpatches = wpsum_te(vid,scores_s,inds)#.view(scores_s.shape[0],-1)
    wpatches = rearrange(wpatches,'b H q pt c h w -> b q 1 pt (H c) h w')
    vid_te,vidz = fold(wpatches)
    vid_te = vid_te / vidz

    # -- ground-truth --
    vid_gt = wpsum_gt(vid,scores_s,inds)

    # -- compare --
    tol = 1e-5
    error = th.abs(vid_gt - vid_te).mean().item()
    if error > tol: print(error)
    assert error < tol

    tol = 1e-6
    error = th.abs(vid_gt - vid_te).max().item()
    if error > tol: print(error)
    assert error < tol


def test_identity(ps,pt,ws,wt,stride0,stride1,dilation,nheads,k):

    # -- init vars --
    use_atomic = True
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    reflect_bounds = True
    use_adj = False # both pass

    # -- load data --
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    vid = get_data(dnames)

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)

    # -- init xsearch --
    dist_type = "l2"
    sch = stnls.search
    search = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                dist_type=dist_type,dilation=dilation,
                                stride0=stride0, stride1=stride1,
                                reflect_bounds=reflect_bounds,anchor_self=True,
                                use_adj=use_adj)

    # -- init our inner product --
    wpsum = stnls.reducer.WeightedPatchSum(ps, pt, dilation=dilation, use_adj=use_adj,
                                           reflect_bounds=reflect_bounds,
                                           use_atomic=use_atomic)
    fold = stnls.iFoldz(vid.shape,stride=stride0,dilation=dilation,
                        use_adj=use_adj,reflect_bounds=reflect_bounds,
                        device=vid.device)

    # -- run search & wpsum --
    scores,inds = search(vid,vid,flows.fflow,flows.bflow)
    scores_s = softmax(-scores*10,dim=-1)
    scores_s[...,0] = 1
    scores_s[...,1:] = 0
    wpatches = wpsum(vid,scores_s,inds)#.view(scores_s.shape[0],-1)
    wpatches = rearrange(wpatches,'b H q pt c h w -> b q 1 pt (H c) h w')
    avid,vidz = fold(wpatches)
    avid = avid / vidz

    # -- compare --
    tol = 1e-5
    error = th.abs(avid - vid).mean().item()
    if error > tol: print(error)
    assert error < tol

    tol = 1e-6
    error = th.abs(avid - vid).max().item()
    if error > tol: print(error)
    assert error < tol



def test_score_backward(ps,pt,ws,wt,stride0,stride1,dilation,nheads,k):

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    reflect_bounds = False
    use_adj = True
    use_atomic = True

    # -- load data --
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    vid = get_data(dnames)

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)

    # -- init xsearch --
    dist_type = "l2"
    sch = stnls.search
    search = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                dist_type=dist_type,dilation=dilation,
                                stride0=stride0, stride1=stride1,
                                reflect_bounds=reflect_bounds,anchor_self=True,
                                use_adj=use_adj)

    # -- init our inner product --
    wpsum = stnls.reducer.WeightedPatchSum(ps, pt, dilation=dilation, use_adj=use_adj,
                                           reflect_bounds=reflect_bounds,
                                           use_atomic=use_atomic)

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
                                                 use_adj=use_adj,pt=pt,dilation=dilation,
                                                 reflect_bounds=reflect_bounds)

    # -- confirm fwd --
    tol = 1e-5
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
        if error > tol_mean: print("mean error: ",error)
        assert error < tol_mean

        error = diff[args].max().item()
        if error > tol_max: print("max error: ",error)
        assert error < tol_max

# @pytest.mark.slow
def test_vid_backward(ps,pt,ws,wt,stride0,stride1,dilation,nheads,k):

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    reflect_bounds = False
    use_adj = True
    use_atomic = True

    # -- load data --
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    ext = "jpg"
    vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)/255.
    vid = vid.to(device)

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)

    # -- init xsearch --
    dist_type = "l2"
    sch = stnls.search
    search = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                dist_type=dist_type,dilation=dilation,
                                stride0=stride0, stride1=stride1,
                                reflect_bounds=reflect_bounds,anchor_self=True,
                                use_adj=use_adj)

    # -- init our inner product --
    wpsum = stnls.reducer.WeightedPatchSum(ps, pt, dilation=dilation, use_adj=use_adj,
                                           reflect_bounds=reflect_bounds,
                                           use_atomic=use_atomic)

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
                                                 use_adj=use_adj,pt=pt,dilation=dilation,
                                                 reflect_bounds=reflect_bounds)

    # -- confirm fwd --
    tol = 1e-5
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
        if error > tol_mean: print("mean error: ",error)
        assert error < tol_mean

        error = diff[args].max().item()
        if error > tol_max: print("max error: ",error)
        assert error < tol_max
