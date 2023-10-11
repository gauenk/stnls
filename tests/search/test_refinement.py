
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
# from stnls.utils.inds import get_batching_info

# -- test func --
from torch.nn.functional import fold,unfold,pad
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/non_local_search")

def pytest_generate_tests(metafunc):
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)
    test_lists = {"ws":[3],"wt":[1],"k":[-1],"wr":[3],"kr":[-1],
                  "ps":[1],"stride0":[4],"stride1":[1],"dilation":[1],
                  "self_action":[None],"nheads":[1],"seed":[0],
                  "dist_type":["l2"],"itype":["float"]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def test_fwd(ws,wt,wr,kr,k,ps,stride0,stride1,dilation,
             self_action,nheads,dist_type,itype,seed):
    """

    Test the CUDA code with torch code

    Forward Pass

    """


    # -- init vars --
    dil = dilation
    pt = 1
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    reflect_bounds = False
    use_adj = False
    self_action = None
    full_ws = False
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]

    # -- load data --
    vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid.to(device)[:,:5,:3,::2,::2].contiguous()
    vid = repeat(vid,'b t c h w -> b t (r c) h w',r=12)[:,:32].contiguous()
    vid /= vid.max()

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)
    fflow = 10*th.randn_like(flows.fflow)
    bflow = 10*th.randn_like(flows.bflow)

    # -- exec fold fxns --
    search = stnls.search.NonLocalSearch(ws, wt, ps, k, nheads,
                                         dilation=dil,stride0=stride0, stride1=stride1,
                                         reflect_bounds=reflect_bounds,full_ws=full_ws,
                                         self_action=self_action,use_adj=use_adj,
                                         dist_type=dist_type,itype=itype)
    refine = stnls.search.RefineSearch(ws, wt, wr, k, kr, ps, nheads,
                                       dilation=dil,stride0=stride0, stride1=stride1,
                                       reflect_bounds=reflect_bounds,full_ws=full_ws,
                                       self_action=self_action,use_adj=use_adj,
                                       dist_type=dist_type,
                                       itype_fwd=itype,itype_bwd=itype)

    # -- test api --
    dists_gt,inds_gt = search(vid,vid,fflow,bflow)
    th.cuda.synchronize()
    dists_te,inds_te = refine(vid,vid,inds_gt)

    # -- compare --
    assert th.allclose(dists_te,dists_gt,1e-3,1e-3,equal_nan=True).item()


# @pytest.mark.slow
def test_bwd(ws,wt,wr,kr,ps,stride0,stride1,dilation,self_action,
                  k,nheads,dist_type,itype,seed):
    """

    Test the CUDA code with torch code

    Forward Pass

    """


    # -- init vars --
    dil = dilation
    pt = 1
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    reflect_bounds = True
    use_adj = False
    self_action = None
    full_ws = True
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]

    # -- load data --
    vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid.to(device)[:1,:5,:3,::4,::4].contiguous()
    vid = repeat(vid,'b t c h w -> b t (r c) h w',r=12)[:,:,:3].contiguous()
    vid /= vid.max()
    vid0 = th.rand_like(vid)-0.5
    vid1 = th.rand_like(vid)-0.5
    T = vid.shape[1]

    # -- init for grads --
    vid0_srch,vid1_srch = vid0.clone(),vid1.clone()
    vid0.requires_grad_(True)
    vid1.requires_grad_(True)

    # -- compute flow --
    flows = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)
    fflow = (th.rand_like(flows.fflow)/(2.*T)+0.1)
    bflow = (th.rand_like(flows.bflow)/(2.*T)+0.1)
    # fflow = 2*(th.rand_like(flows.fflow)-0.5)
    # bflow = 2*(th.rand_like(flows.bflow)-0.5)


    # -- exec fold fxns --
    k_search = 5
    search = stnls.search.NonLocalSearch(ws, wt, ps, k_search, nheads,
                                         dilation=dil,stride0=stride0, stride1=stride1,
                                         reflect_bounds=reflect_bounds,full_ws=full_ws,
                                         self_action="remove",use_adj=use_adj,
                                         dist_type=dist_type,itype=itype)
    refine = stnls.search.RefineSearch(ws, wt, wr, k, kr, ps, nheads,
                                       dilation=dil,stride0=stride0, stride1=stride1,
                                       reflect_bounds=reflect_bounds,full_ws=full_ws,
                                       self_action=self_action,use_adj=use_adj,
                                       dist_type=dist_type,
                                       itype_fwd=itype,itype_bwd=itype)

    # -- test api --
    srch_dists,srch_inds = search(vid0_srch,vid1_srch,fflow,bflow)
    # srch_inds = srch_inds.detach()[...,ws*ws:ws*ws+5,:] # skip self
    srch_inds = srch_inds.requires_grad_(True)
    th.cuda.synchronize()
    # print(srch_dists[0,0,:2,:2,:2,:2])
    # print(srch_inds[0,0,:2,:2,:2,:2])

    # -- autograd --
    fxn = lambda vid0: refine(vid0,vid1,srch_inds)[0]
    assert gradcheck_skipnan(fxn,vid0, atol=1e-02)
    fxn = lambda vid0: refine(vid0,vid1,srch_inds)[0]
    assert gradcheck_skipnan(fxn,vid1, atol=1e-02)

    # -- autograd check for indices --
    srch_inds_t =  srch_inds[...,[0]]
    srch_inds_sp =  srch_inds[...,1:].requires_grad_(True)
    def fxn(srch_inds_sp):
        srch_inds = th.cat([srch_inds_t,srch_inds_sp],-1).requires_grad_(True)
        return refine(vid0,vid1,srch_inds)[0]
    assert gradcheck_skipnan(fxn, srch_inds_sp, atol=1e-02)

    if itype == "float":
        def fxn(srch_inds_sp):
            srch_inds = th.cat([srch_inds_t,srch_inds_sp],-1).requires_grad_(True)
            return refine(vid0,vid1,srch_inds)[1]
        assert gradcheck_skipnan(fxn, srch_inds_sp, atol=1e-02)
        # th.autograd.gradcheck(fxn, srch_inds_sp, eps=1e-2,
        #                       atol=1e-2, nondet_tol=1e-7, raise_exception=True)


def gradcheck_skipnan(fxn,inputs, rtol=1e-05, atol=1e-08):
    num,ana = get_gradcheck_pair(fxn,inputs)
    args = th.where(th.logical_and(~th.isnan(num),num.abs()>0))
    # args = th.where(~th.isnan(num))
    # print(num[args])
    # print(ana[args])
    return th.allclose(num[args],ana[args],atol=atol,rtol=rtol)

def get_gradcheck_pair(fxn,inputs):
    from torch.autograd.gradcheck import get_numerical_jacobian,get_analytical_jacobian
    from torch.autograd.gradcheck import _get_numerical_jacobian
    from torch.autograd.gradcheck import _check_analytical_jacobian_attributes
    num = _get_numerical_jacobian(fxn, (inputs,),
                                  eps=1e-3, is_forward_ad=False)[0][0]
    out = fxn(inputs)
    ana = _check_analytical_jacobian_attributes((inputs,), out, 1e-7, False)[0]
    return num,ana

