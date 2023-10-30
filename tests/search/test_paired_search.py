
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
from stnls.testing import check_shuffled_inds

# -- test func --
from torch.nn.functional import fold,unfold,pad
from torchvision.transforms.functional import center_crop
from torchvision.utils import make_grid,save_image

# -- paths --
SAVE_DIR = Path("./output/tests/paired_search")

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def pytest_generate_tests(metafunc):
    # seed = 123
    # th.manual_seed(seed)
    # np.random.seed(seed)
    test_lists = {"ws":[3],"wt":[1],"k":[-1],"pt":[1],
                  "ps":[3],"stride0":[1],"stride1":[1],"dilation":[1],
                  "self_action":["anchor_each",None],"nheads":[1],"seed":[0],
                  "dist_type":["l2","prod"],"itype":["int","float"],
                  # "dist_type":["l2"],"itype":["float"],
                  "reflect_bounds":[True]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def test_fwd(ws,wt,ps,pt,stride0,stride1,dilation,
             nheads,self_action,dist_type,seed,itype,reflect_bounds):

    # -- get args --
    device = "cuda:0"
    set_seed(seed)
    W_t = 2*wt+1
    k = W_t*ws*ws

    # -- load data --
    B,T,F,H,W = 1,5,3,16,16
    vid = th.ones((B,T,F,H,W),device=device).float()
    vid0 = th.randn_like(vid)-0.5
    vid1 = th.randn_like(vid)

    # -- load flows --
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    flows = th.ones((B,1,T,W_t-1,2,nH,nW)).cuda()#/2.
    flows = th.rand_like(flows)/2.+th.randint_like(flows,-2,2)+0.2
    # flows = th.zeros_like(flows)

    # -- exec fold fxns --
    sch = stnls.search
    search_gt = sch.NonLocalSearch(ws, wt, ps, k, nheads, dist_type=dist_type,
                                   dilation=dilation,stride0=stride0, stride1=stride1,
                                   reflect_bounds=reflect_bounds,full_ws=True,
                                   self_action=self_action,
                                   topk_mode="each",itype=itype)
    search_te = sch.PairedSearch(ws, ps, k, nheads, dist_type=dist_type,
                                 dilation=dilation,stride0=stride0, stride1=stride1,
                                 reflect_bounds=reflect_bounds,full_ws=True,
                                 self_action=self_action,itype=itype)

    # -- [groundtruth] search --
    dists_gt,inds_gt = search_gt(vid0,vid1,flows)

    # -- [testing] search --
    dists_te,inds_te = search_te.paired_vids(vid0, vid1, flows, wt)

    # -- compare --
    atol,rtol = 1e-2,1e-5
    assert th.allclose(dists_gt,dists_te,atol=atol,rtol=rtol)
    try:
        assert th.allclose(inds_gt,inds_te,atol=atol,rtol=rtol)
    except:
        assert check_shuffled_inds(inds_gt,inds_te)

def test_bwd(ws,wt,k,ps,stride0,stride1,dilation,
             nheads,self_action,dist_type,seed,itype,reflect_bounds):

    # -- get args --
    dil = dilation
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    pt = 1
    seed = 234
    device = "cuda:0"
    reflect_bounds = True
    use_adj = False
    set_seed(seed)

    # -- load data --
    B,T,F,H,W = 2,10,16,16,7
    vid = th.ones((B,T,F,H,W),device=device).float()
    vid0 = th.randn_like(vid)
    vid1 = th.randn_like(vid)

    # -- load flows --
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    W_t = 2*wt+1
    flows = th.ones((B,1,T,W_t-1,2,nH,nW)).cuda()/2.
    flows = th.rand_like(flows)/2.+th.randint_like(flows,-2,2)+0.2
    # flows = th.zeros_like(flows)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape

    # -- exec fold fxns --
    sch = stnls.search
    search_gt = sch.NonLocalSearch(ws, wt, ps, k, nheads, dist_type=dist_type,
                                   dilation=dil,stride0=stride0, stride1=stride1,
                                   reflect_bounds=reflect_bounds,full_ws=True,
                                   topk_mode="each",self_action=self_action,
                                   use_adj=use_adj,itype=itype)
    search_te = sch.PairedSearch(ws, ps, k, nheads, dist_type=dist_type,
                                 dilation=dil,stride0=stride0, stride1=stride1,
                                 reflect_bounds=reflect_bounds,full_ws=True,
                                 self_action=self_action,use_adj=use_adj,itype=itype)

    # -- [groundtruth] search --
    vid0_gt = vid0.clone().requires_grad_(True)
    vid1_gt = vid1.clone().requires_grad_(True)
    flows_gt = flows.clone().requires_grad_(True)
    dists_gt,inds_gt = search_gt(vid0_gt,vid1_gt,flows_gt)

    # -- [testing] search --
    vid0_te = vid0.clone().requires_grad_(True)
    vid1_te = vid1.clone().requires_grad_(True)
    flows_te = flows.clone().requires_grad_(True)
    dists_te,inds_te = search_te.paired_vids(vid0_te, vid1_te, flows_te, wt)

    # -- compare --
    atol,rtol = 1e-2,1e-5
    assert th.allclose(dists_gt,dists_te,atol=atol,rtol=rtol)
    assert th.allclose(inds_gt,inds_te,atol=atol,rtol=rtol)

    # -- backprop dists --
    dists_grad = th.randn_like(dists_gt)
    th.autograd.backward(dists_gt,dists_grad,retain_graph=True)
    th.autograd.backward(dists_te,dists_grad,retain_graph=True)

    # -- vid+flow grads --
    atol,rtol = 1e-2,1e-3
    _grads_gt = [vid0_gt.grad,vid1_gt.grad,]
    _grads_te = [vid0_te.grad,vid1_te.grad,]
    if itype == "float":
        _grads_gt += [flows_gt.grad]
        _grads_te += [flows_te.grad]
    for idx,(grads_gt,grads_te) in enumerate(zip(_grads_gt,_grads_te)):
        assert th.allclose(grads_gt,grads_te,atol=atol,rtol=rtol)

    if itype == "float":
        # -- backprop inds --
        flows_te.grad[...] = 0
        flows_gt.grad[...] = 0
        inds_grad = th.randn_like(inds_gt)
        th.autograd.backward(inds_gt,inds_grad,retain_graph=True)
        th.autograd.backward(inds_te,inds_grad,retain_graph=True)

        # -- flow grads --
        atol,rtol = 1e-2,1e-5
        _grads_gt = [flows_gt.grad]
        _grads_te = [flows_te.grad]
        for idx,(grads_gt,grads_te) in enumerate(zip(_grads_gt,_grads_te)):
            assert th.allclose(grads_gt,grads_te,atol=atol,rtol=rtol)


