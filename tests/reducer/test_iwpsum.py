"""

Weighted-Patch (Inplace) Sum "Heads"


Verbose Psuedo-Code:

   patches_i = unfold_k(b2,nlInds_cu).type(th.float64)
   patches_i = rearrange(patches_i,'n k 1 c h w -> n k (c h w)')
   zpatches = []
   for ki in range(k):
      yi = softmax(dists[ki])
      zi = th.sum(yi * patches_i,1).type(th.float32) # n (c h w), this code!
      zpatches.append(zi)
   zpatches = th.stack(zpatches)

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


# -- test func --
from torch.nn.functional import fold,unfold,pad,softmax,log_softmax
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/reducer/iwpsum/")

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
    test_lists = {"ps":[7],"pt":[1],"stride0":[4],"stride1":[1],
                  "dilation":[1],"wt":[3],"k":[5],
                  "ws":[10],"nheads":[1],"batchsize":[-1],
                  "reflect_bounds":[True]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def test_forward(ps,pt,ws,wt,k,stride0,stride1,dilation,nheads,batchsize):

    # -- init vars --
    use_atomic = False
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    reflect_bounds = True
    use_adj = False

    # -- load data --
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    vid = get_data(dnames)

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
                                pt=pt,dist_type=dist_type,dilation=dilation,
                                stride0=stride0, stride1=stride1,
                                reflect_bounds=reflect_bounds,anchor_self=True,
                                use_adj=use_adj)

    # -- init our inner product --
    wpsum_gt = stnls.reducer.FoldedWeightedPatchSum(ps, stride0, -1,
                                                    pt, dilation=dilation,
                                                    use_adj=use_adj,
                                                    reflect_bounds=reflect_bounds,
                                                    use_atomic=use_atomic)
    wpsum_te = stnls.reducer.InplaceWeightedPatchSum(ps, batchsize,
                                                     pt, dilation=dilation,
                                                     use_adj=use_adj,
                                                     reflect_bounds=reflect_bounds)

    # -- run search --
    scores,inds = search(vid,vid,flows.fflow,flows.bflow)
    scores_s = softmax(-scores,dim=-1)
    scores_s = th.randn_like(scores_s)
    scores_s = softmax(-scores,dim=-1)

    # -- testing --
    vid_te = wpsum_te(vid,scores_s,inds)

    # -- ground-truth --
    vid_gt = wpsum_gt(vid,scores_s,inds)

    # -- viz --
    # print(vid_te[0,0,0,:3,:3])
    # print(vid_gt[0,0,0,:3,:3])
    # print((vid_te/vid_gt)[0,0,0,:3,:3])
    print(vid_te[0,0,0,-8:,-8:])
    print(vid_gt[0,0,0,-8:,-8:])
    print((vid_te/vid_gt)[0,0,0,-8:,-8:])

    diff = th.abs(vid_gt - vid_te)
    print(th.where(diff > 0.1))
    print(vid_gt[th.where(diff > 0.1)])
    print(vid_te[th.where(diff > 0.1)])

    # -- compare --
    tol = 1e-4
    error = th.abs(vid_gt - vid_te).mean().item()
    if error > tol: print(error)
    assert error < tol

    tol = 1e-2
    error = th.abs(vid_gt - vid_te).max().item()
    if error > tol: print(error)
    assert error < tol


def test_score_backward(ps,pt,ws,wt,k,stride0,stride1,dilation,nheads,batchsize):

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    reflect_bounds = False
    use_adj = False
    use_atomic = True

    # -- load data --
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    vid = get_data(dnames)

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape

    # -- init --
    dist_type = "l2"
    sch = stnls.search
    search = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                pt=pt,dist_type=dist_type,dilation=dilation,
                                stride0=stride0, stride1=stride1,
                                reflect_bounds=reflect_bounds,anchor_self=True,
                                use_adj=use_adj)
    wpsum_gt = stnls.reducer.FoldedWeightedPatchSum(ps, stride0, -1,
                                                    pt, dilation=dilation,
                                                    use_adj=use_adj,
                                                    reflect_bounds=reflect_bounds,
                                                    use_atomic=use_atomic)
    wpsum_te = stnls.reducer.InplaceWeightedPatchSum(ps, batchsize,
                                                     pt, dilation=dilation,
                                                     use_adj=use_adj,
                                                     reflect_bounds=reflect_bounds)

    # -- run search --
    scores,inds = search(vid,vid,flows.fflow,flows.bflow)
    scores_s = softmax(-scores*10,dim=-1)

    # -- forwards --
    scores_te = scores_s.clone()
    scores_te.requires_grad_(True)
    avid_te = wpsum_te(vid,scores_te,inds)

    # -- ground-truth --
    scores_gt = scores_s.clone()
    scores_gt.requires_grad_(True)
    avid_gt = wpsum_te(vid,scores_gt,inds)

    # -- confirm fwd --
    tol = 1e-7
    error = th.abs(avid_te - avid_gt).mean().item()
    if error > tol: print(error)
    assert error < tol

    # -- backward passes --
    avid_grad = th.rand_like(avid_te)
    th.autograd.backward(avid_te,avid_grad)
    th.autograd.backward(avid_gt,avid_grad)

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
def test_vid_backward(ps,pt,ws,wt,k,stride0,stride1,dilation,nheads,batchsize):

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    reflect_bounds = False
    use_adj = False
    use_atomic = True

    # -- load data --
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    vid = get_data(dnames)

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape

    # -- init --
    dist_type = "l2"
    sch = stnls.search
    search = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                pt=pt,dist_type=dist_type,dilation=dilation,
                                stride0=stride0, stride1=stride1,
                                reflect_bounds=reflect_bounds,anchor_self=True,
                                use_adj=use_adj)
    wpsum_gt = stnls.reducer.FoldedWeightedPatchSum(ps, stride0, -1,
                                                    pt, dilation=dilation,
                                                    use_adj=use_adj,
                                                    reflect_bounds=reflect_bounds,
                                                    use_atomic=use_atomic)
    wpsum_te = stnls.reducer.InplaceWeightedPatchSum(ps, batchsize,
                                                     pt, dilation=dilation,
                                                     use_adj=use_adj,
                                                     reflect_bounds=reflect_bounds)

    # -- run search --
    scores,inds = search(vid,vid,flows.fflow,flows.bflow)
    scores_s = softmax(-scores*100,dim=-1)

    # -- require grads  --
    vid_te = vid.clone()
    vid_te = vid_te.requires_grad_(True)
    vid_gt = vid.clone()
    vid_gt = vid_gt.requires_grad_(True)

    # -- testing --
    avid_te = wpsum_te(vid_te,scores_s,inds)

    # -- ground-truth --
    avid_gt = wpsum_te(vid_gt,scores_s,inds)

    # -- confirm fwd --
    tol = 1e-7
    error = th.abs(avid_te - avid_gt).mean().item()
    if error > tol: print(error)
    assert error < tol

    # -- backward passes --
    avid_grad = th.rand_like(avid_te)
    th.autograd.backward(avid_te,avid_grad)
    th.autograd.backward(avid_gt,avid_grad)

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

def test_forward_v2(ps,pt,ws,wt,k,stride0,stride1,dilation,nheads,batchsize):

    # -- init vars --
    use_atomic = False
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

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape

    # -- init xsearch --
    dist_type = "l2"
    sch = stnls.search
    search = sch.NonLocalSearch(ws, wt, ps, k, nheads,
                                pt=pt,dist_type=dist_type,dilation=dilation,
                                stride0=stride0, stride1=stride1,
                                reflect_bounds=reflect_bounds,anchor_self=True,
                                use_adj=use_adj)
    wpsum_te = stnls.reducer.InplaceWeightedPatchSum(ps, batchsize,
                                                     pt, dilation=dilation,
                                                     use_adj=use_adj,
                                                     reflect_bounds=reflect_bounds)

    # -- init our inner product --
    wpsum = stnls.reducer.WeightedPatchSum(ps, pt, dilation=dilation,
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
    wpatches = wpsum(vid,scores_s,inds)#.view(scores_s.shape[0],-1)
    wpatches = rearrange(wpatches,'b H q pt c h w -> b q 1 pt (H c) h w')
    vid_gt,vidz = fold(wpatches)
    vid_gt = vid_gt / vidz

    # -- testing --
    vid_te = wpsum_te(vid,scores_s,inds)

    # -- compare --
    tol = 1e-4
    error = th.abs(vid_te - vid_gt).mean().item()
    if error > tol: print(error)
    assert error < tol

    tol = 1e-2
    error = th.abs(vid_te - vid_gt).max().item()
    if error > tol: print(error)
    assert error < tol

