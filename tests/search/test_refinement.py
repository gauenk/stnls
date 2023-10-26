
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


def gradcheck_skip_nan_unstable(fxn,inputs, rtol=1e-05, atol=1e-08,
                                nreps=3, num_eps=5e-4, unstable_eps=1e-2):
    num = get_num_jacobian_skip_unstable(fxn,inputs,eps=num_eps,
                                         nreps=nreps,unstable_eps=unstable_eps)
    ana = get_ana_jacobian(fxn,inputs)
    args = th.where(th.logical_and(~th.isnan(num),num.abs()>0))
    args1 = th.where(th.abs(num[args]-ana[args])>1e-2)[0]
    # print("ana: ",ana[47,573:575])
    # print(num[:5,:5])
    # print(ana[:5,:5])
    # print(num[-5:,-5:])
    # print(ana[-5:,-5:])
    # # print(num.shape)
    # print(num[args][args1][:20])
    # print(ana[args][args1][:20])
    # print([args[i][args1] for i in range(2)])
    return th.allclose(num[args],ana[args],atol=atol,rtol=rtol)

def gradcheck_skipnan(fxn,inputs, rtol=1e-05, atol=1e-08, nreps=1, num_eps=5e-4):
    num = get_num_jacobian(fxn,inputs,eps=num_eps,nreps=nreps)
    ana = get_ana_jacobian(fxn,inputs)
    args = th.where(th.logical_and(~th.isnan(num),num.abs()>0))
    args1 = th.where(th.abs(num[args]-ana[args])>1e-2)[0]
    # print(num[-5:,-5:])
    # print(ana[-5:,-5:])
    # print(num.shape)
    # print(num[args][args1][:20])
    # print(ana[args][args1][:20])
    # print([args[i][args1] for i in range(2)])
    return th.allclose(num[args],ana[args],atol=atol,rtol=rtol)

def get_num_jacobian_skip_unstable(fxn,inputs,eps=1e-3,nreps=1,unstable_eps=1e-2):
    from torch.autograd.gradcheck import _get_numerical_jacobian
    nums = []
    for i in range(nreps):
        eps_i = eps * (1 + i*eps)
        num = _get_numerical_jacobian(fxn, (inputs,),
                                      eps=eps_i, is_forward_ad=False)[0][0]
        nums.append(num)

    delta = th.zeros_like(nums[0])
    for i in range(nreps):
        # print(nums[i][47,573:575])
        for j in range(nreps):
            if i >= j: continue
            # print(i,j)
            delta += th.abs(nums[i] - nums[j])
    # print(delta)
    # print(delta[~th.isnan(delta)].min(),delta[~th.isnan(delta)].max())
    # print("Percentage unstable: ",100*th.mean(1.*(delta > unstable_eps)).item())
    unstable = th.where(th.logical_or(delta > unstable_eps,th.isnan(delta)))
    num = th.mean(th.stack(nums),dim=0)
    num[unstable] = th.nan
    # print(num)
    # print(nums[0])
    return num

def get_num_jacobian(fxn,inputs,eps=1e-3,nreps=1):
    from torch.autograd.gradcheck import _get_numerical_jacobian
    num = _get_numerical_jacobian(fxn, (inputs,),
                                  eps=eps, is_forward_ad=False)[0][0]
    for i in range(nreps-1):
        num += get_num_jacobian(fxn,inputs,eps=eps)
    num /= nreps
    return num

def get_ana_jacobian(fxn,inputs):
    from torch.autograd.gradcheck import _check_analytical_jacobian_attributes
    out = fxn(inputs)
    ana = _check_analytical_jacobian_attributes((inputs,), out, 1e-7, False)[0]
    return ana

def get_gradcheck_pair(fxn,inputs,eps=1e-3):
    num = get_num_jacobian(fxn,inputs,eps=1e-3)
    ana = get_ana_jacobian(fxn,inputs)
    return num,ana

def pytest_generate_tests(metafunc):
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)
    test_lists = {"ws":[3],"wt":[1],"k":[-1],"wr":[1],"kr":[-1],
                  "ps":[3],"stride0":[1],"stride1":[1],"dilation":[1],
                  "self_action":["anchor_each"],"nheads":[1],"seed":[0],
                  "dist_type":["prod","l2"],"itype":["int","float"],
                  # "dist_type":["l2"],"itype":["float"],
                  "reflect_bounds":[True]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def test_refine_fwd(ws,wt,wr,kr,k,ps,stride0,stride1,dilation,
                    nheads,dist_type,itype,seed,reflect_bounds):
    """

    Test the CUDA code with torch code

    Forward Pass

    """


    # -- init vars --
    self_action = None
    dil = dilation
    pt = 1
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    use_adj = False
    self_action = None
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
                                         reflect_bounds=reflect_bounds,full_ws=True,
                                         self_action=self_action,use_adj=use_adj,
                                         dist_type=dist_type,itype=itype)
    refine = stnls.search.RefineSearch(ws, wt, 1, -1, kr, ps, nheads,
                                       dilation=dil,stride0=stride0, stride1=stride1,
                                       reflect_bounds=reflect_bounds,full_ws=True,
                                       self_action=self_action,use_adj=use_adj,
                                       dist_type=dist_type,
                                       itype_fwd=itype,itype_bwd=itype)

    # -- test api --
    dists_gt,inds_gt = search(vid,vid,fflow,bflow)
    th.cuda.synchronize()
    dists_te,inds_te = refine(vid,vid,inds_gt)
    # print(dists_gt.shape,dists_te.shape)

    # -- compare --
    assert th.allclose(dists_te,dists_gt,1e-3,1e-3,equal_nan=True)

def int_spaced_vid(B,T,F,H,W):
    device = "cuda:0"
    dtype = th.float32
    grid_y, grid_x = th.meshgrid(th.arange(0, H, dtype=dtype, device=device),
                                 th.arange(0, W, dtype=dtype, device=device))
    grid = th.stack((grid_x, grid_y), 0).float()[None,:]  # 2, W(x), H(y)
    vid = []
    for ti in range(T):
        g0 = grid[:,[0]].repeat(B,F,1,1)/W-0.5
        g1 = grid[:,[1]].repeat(B,F,1,1)/H-0.5
        # g0 += th.rand_like(g0)
        # g1 += th.rand_like(g1)
        tN = (ti+1)*th.ones_like(g0)
        vid.append(g0*g1*tN) # noise less than int
    vid = th.stack(vid,1)
    return vid

# @pytest.mark.slow
def test_refine_noshuffle_bwd(ws,wt,wr,kr,ps,stride0,stride1,dilation,
                              k,nheads,dist_type,itype,seed,reflect_bounds):
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
    use_adj = False
    self_action = None
    full_ws = True
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    set_seed(seed)

    # -- shapes --
    W_t = 2*wt+1
    k,kr = W_t*ws*ws,-1
    HD,K = nheads,k

    # -- load data --
    B,T,F,H,W = 1,3,1,8,8
    W_t = 2*wt+1
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    vid0 = int_spaced_vid(B,T,F,H,W)
    vid0 = int_spaced_vid(B,T,F,H,W)
    vid1 = int_spaced_vid(B,T,F,H,W)
    vid0 = th.rand_like(vid0)/2.+0.2
    vid1 = th.rand_like(vid0)/2.+0.2

    # -- init for grads --
    vid0_srch,vid1_srch = vid0.clone(),vid1.clone()
    vid0.requires_grad_(True)
    vid1.requires_grad_(True)

    # -- exec fold fxns --
    wr = 1
    refine = stnls.search.RefineSearch(ws, wt, wr, -1, kr, ps, nheads,
                                       dilation=dil,stride0=stride0, stride1=stride1,
                                       reflect_bounds=reflect_bounds,full_ws=full_ws,
                                       self_action=self_action,use_adj=use_adj,
                                       dist_type=dist_type,
                                       itype_fwd=itype,itype_bwd=itype,topk_mode="all")

    # -- create inds --
    srch_inds = th.ones((B,HD,T,nH,nW,K,3))+0.1
    srch_inds = th.rand_like(srch_inds)/2.+1.1
    tgrid = th.arange(0,T).view(1,1,T,1,1,1)
    srch_inds[...,0] = th.randint(0,T,size=srch_inds[...,0].shape)-tgrid
    srch_inds[...,1:] = th.rand_like(srch_inds[...,1:])/2.+0.2
    # srch_inds[...,1:] = -srch_inds[...,1:]
    not_int = th.all(th.abs(srch_inds[...,1:].round() - srch_inds[...,1:])>1e-5).item()
    assert not_int,"Gradcheck only works _not_ near an int."
    srch_inds = srch_inds.to(vid0.device)
    # srch_inds = srch_inds.requires_grad_(True)

    # -- run refinement --
    ref_dists,ref_inds = refine(vid0,vid1,srch_inds)

    # -- autograd --
    fxn = lambda vid0: refine(vid0,vid1,srch_inds)[0]
    # assert gradcheck_skip_nan_unstable(fxn,vid0, atol=1e-02, num_eps=1e-5)
    assert gradcheck_skipnan(fxn,vid0, atol=1e-02, num_eps=1e-3)
    fxn = lambda vid1: refine(vid0,vid1,srch_inds)[0]
    assert gradcheck_skipnan(fxn,vid1, atol=1e-02, num_eps=1e-3)
    # assert gradcheck_skip_nan_unstable(fxn,vid1, atol=1e-02, num_eps=1e-5)

    # -- autograd check for indices --
    if itype == "float":
        srch_inds_t =  srch_inds[...,[0]]
        srch_inds_sp =  srch_inds[...,1:].requires_grad_(True)
        def fxn(srch_inds_sp):
            srch_inds = th.cat([srch_inds_t,srch_inds_sp],-1).requires_grad_(True)
            return refine(vid0,vid1,srch_inds)[0]
        # assert gradcheck_skipnan(fxn, srch_inds_sp, atol=1e-02, num_eps=1e-5)
        # assert gradcheck_skip_nan_unstable(fxn, srch_inds_sp, atol=1e-02,
        #                                    nreps=3, num_eps=1e-3)

        def fxn(srch_inds_sp):
            srch_inds = th.cat([srch_inds_t,srch_inds_sp],-1).requires_grad_(True)
            return refine(vid0,vid1,srch_inds)[1]
        # assert gradcheck_skipnan(fxn, srch_inds_sp, atol=1e-02, num_eps=1e-5)
        assert gradcheck_skip_nan_unstable(fxn, srch_inds_sp, atol=1e-02,
                                           nreps=3, num_eps=1e-3)



def test_anchor_fwd(ws,wt,wr,ps,stride0,stride1,dilation,
                    nheads,dist_type,itype,seed,reflect_bounds):

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
    use_adj = False
    full_ws = False
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    topk_mode = "each"
    kr = -1

    # -- load data --
    vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid.to(device)[:,:5,:3,::2,::2].contiguous()
    vid = repeat(vid,'b t c h w -> b t (r c) h w',r=12)[:,:32].contiguous()
    vid0 = th.rand_like(vid)-0.5
    vid1 = th.rand_like(vid)-0.5

    # -- compute flow --
    B,T,F,H,W = vid.shape
    W_t = 2*wt+1
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    flows = 2*th.rand((B,T,W_t-1,2,nH,nW)).to(vid0.device)

    # -- exec fold fxns --
    k0 = 5
    search0 = stnls.search.NonLocalSearch(ws, wt, ps, k0, nheads,
                                          dilation=dil,stride0=stride0, stride1=stride1,
                                          reflect_bounds=reflect_bounds,full_ws=True,
                                          self_action=None,use_adj=use_adj,
                                          dist_type=dist_type,topk_mode="each",
                                          itype=itype)
    k = 1
    refine0 = stnls.search.RefineSearch(ws, wt, wr, k, kr, ps, nheads,
                                        dilation=dil,stride0=stride0, stride1=stride1,
                                        reflect_bounds=reflect_bounds,full_ws=False,
                                        self_action="anchor_each",use_adj=use_adj,
                                        dist_type=dist_type,topk_mode="each",
                                        itype_fwd=itype,itype_bwd=itype)
    k = 1
    refine1 = stnls.search.RefineSearch(ws, wt, wr, k, kr, ps, nheads,
                                        dilation=dil,stride0=stride0, stride1=stride1,
                                        reflect_bounds=reflect_bounds,full_ws=True,
                                        self_action="anchor_each",use_adj=use_adj,
                                        dist_type=dist_type,topk_mode="each",
                                        itype_fwd=itype,itype_bwd=itype)


    # -- exec search --
    HD = nheads
    vshape = (B,HD,T,nH,nW,W_t*k0)
    dists0,inds0 = search0(vid0,vid1,flows)
    dists0,inds0 = dists0.view(vshape),inds0.view(vshape+(3,))

    # -- exec refine --
    dists_r0,inds_r0 = refine0(vid0,vid1,inds0)
    dists_r1,inds_r1 = refine1(vid0,vid1,inds0)
    # print(th.stack([dists0,dists_r0],-1))
    # print(th.stack([inds0,inds_r0],-1))
    # print(th.stack([dists0,dists_r1],-1))
    # args = th.where(th.abs(dists0-dists_r0)>1e-3)
    # print(dists0[args][:10])
    # print(dists_r0[args][:10])

    # -- compare --
    assert th.allclose(dists0,dists_r0,1e-2,1e-3,equal_nan=True)
    assert th.allclose(inds0,inds_r0,1e-3,1e-3,equal_nan=True)
    assert th.allclose(dists0,dists_r1,1e-3,1e-3,equal_nan=True)
    assert th.allclose(inds0,inds_r1,1e-3,1e-3,equal_nan=True)


def test_fwd_topk(ws,wt,wr,ps,stride0,stride1,dilation,dist_type,seed,reflect_bounds):

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
    use_adj = False
    full_ws = True
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    topk_mode = "each"
    itype = "float"
    nheads = 1
    kr = -1

    # -- load data --
    vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid.to(device)[:,:5,:3,:,:].contiguous()
    vid = repeat(vid,'b t c h w -> b t (r c) h w',r=12)[:,:32].contiguous()
    vid0 = th.rand_like(vid)-0.5
    vid1 = th.rand_like(vid)-0.5

    # -- compute flow --
    B,T,F,H,W = vid.shape
    W_t = 2*wt+1
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    flows = 2*th.rand((B,T,W_t-1,2,nH,nW)).to(vid0.device)

    # -- exec fold fxns --
    k0 = W_t*ws*ws
    search = stnls.search.NonLocalSearch(ws, wt, ps, -1, nheads,
                                         dilation=dil,stride0=stride0, stride1=stride1,
                                         reflect_bounds=reflect_bounds,full_ws=True,
                                         self_action=None,use_adj=use_adj,
                                         dist_type=dist_type,topk_mode="all",
                                         itype=itype)
    k = k0*wr*wr
    stride1 = 0.1
    refine = stnls.search.RefineSearch(ws, wt, wr, k, kr, ps, nheads,
                                       dilation=dil,stride0=stride0, stride1=stride1,
                                       reflect_bounds=reflect_bounds,full_ws=True,
                                       self_action=None,use_adj=use_adj,
                                       dist_type=dist_type,topk_mode="all",
                                       itype_fwd=itype,itype_bwd=itype)

    # -- exec --
    _dists,_inds = search(vid0,vid1,flows)
    dists,inds = refine(vid0,vid1,_inds)
    # print(_inds[0,0,0,2,2,:10])
    # print(inds[0,0,0,2,2,:10])

    delta = dists[...,1:] - dists[...,:-1]
    delta = delta[~th.isnan(delta)]
    if dist_type == "l2":
        assert th.all(delta>=0).item()
    else:
        assert th.all(delta<=0).item()

