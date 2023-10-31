
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
from stnls.testing.gradcheck import gradcheck_skipnan,gradcheck_skip_nan_unstable
from stnls.testing import check_shuffled_inds,int_spaced_vid

# -- test func --
from torch.nn.functional import fold,unfold,pad
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/non_local_search")


def pytest_generate_tests(metafunc):
    test_lists = {"ws":[3],"wt":[1],"K":[-1],"wr":[5],"kr":[1.],"pt":[1],
                  "ps":[3],"stride0":[1],"stride1":[1],"dilation":[1],
                  "self_action":[None],"nheads":[1],"seed":[0],
                  "dist_type":["l2"],"itype":["int","float"],
                  "topk_mode":["all"],"reflect_bounds":[True]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def test_fwd_match_refine(ws,wt,wr,kr,K,ps,pt,stride0,stride1,dilation,
                          nheads,dist_type,itype,seed,reflect_bounds):
    # -- init --
    B,HD,T,F,H,W = 1,nheads,3,1,10,10
    W_t = min(T,2*wt+1)
    K = W_t*ws*ws if K <= 0 else K
    self_action = None
    # K = -1
    K_refine = int(K*kr)
    K_each = ws*ws
    assert K == (K_each*W_t)
    # K_refine = -1
    device = "cuda:0"
    set_seed(seed)

    # -- video data --
    vid = th.ones((B,T,HD*F,H,W),device=device)
    vid0 = th.rand_like(vid)#.requires_grad_(True)
    vid1 = vid0.clone()#th.rand_like(vid)#.requires_grad_(True)

    # -- create inds --
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    flows = th.ones((B,HD,T,nH,nW,K,3))+0.1
    flows = th.rand_like(flows)/2.+1.1
    for ti in range(T):
        t_grid = stnls.search.utils.get_time_window_inds(ti,wt,T)
        for _tj in range(2*wt+1):
            tj = t_grid[_tj]
            ks,ke = _tj*K_each,(_tj+1)*K_each
            flows[:,:,ti,:,:,ks:ke,0] = tj-ti
    flows[...,1:] = th.rand_like(flows[...,1:])/2.+2.2
    not_int = th.all(th.abs(flows[...,1:].round() - flows[...,1:])>1e-5).item()
    # flows = th.zeros_like(flows)
    assert not_int,"Gradcheck only works _not_ near an int."
    flows = flows.to(vid0.device)

    # -- exec fold fxns --
    refine_gt = stnls.search.RefineSearch(ws, wt, wr, -1, kr, ps, nheads,
                                          dilation=dilation,
                                          stride0=stride0, stride1=stride1,
                                          reflect_bounds=reflect_bounds,full_ws=True,
                                          self_action=self_action,topk_mode="all",
                                          dist_type=dist_type,itype=itype)
    refine_te = stnls.search.PairedRefine(ws, wr, -1, kr, ps, nheads,
                                          dilation=dilation,
                                          stride0=stride0, stride1=stride1,
                                          reflect_bounds=reflect_bounds,full_ws=True,
                                          self_action=self_action,topk_mode="all",
                                          dist_type=dist_type,itype=itype)

    # -- test api --
    dists_gt,inds_gt = refine_gt(vid0, vid1, flows)
    dists_te,inds_te = refine_te.paired_vids(vid0, vid1, flows, wt)

    # -- compare --
    assert th.allclose(dists_te,dists_gt,1e-3,1e-3,equal_nan=True)

def test_fwd_match_search(ws,wt,kr,ps,pt,stride0,stride1,dilation,
                          self_action,nheads,dist_type,itype,seed,reflect_bounds):
    """

    Test the CUDA code with torch code

    Forward Pass

    """

    # -- init vars --
    device = "cuda:0"
    B,HD,T,F,H,W = 1,nheads,3,1,10,10
    wr = 1
    W_t = min(2*wt+1,T)
    K = W_t*ws*ws
    set_seed(seed)

    # -- load data --
    vid = th.ones((B,T,HD*F,H,W),device=device)
    vid0 = th.rand_like(vid)#.requires_grad_(True)
    vid1 = th.rand_like(vid)#.requires_grad_(True)
    # vid0 = th.rand_like(vid)#.requires_grad_(True)
    # vid1 = vid0.clone()#th.rand_like(vid)#.requires_grad_(True)

    # -- compute flow --
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    flows = th.ones((B,HD,T,W_t-1,2,nH,nW)).cuda()/2.
    flows = th.rand_like(flows)/2.+th.randint_like(flows,-3,3)+0.2
    # flows = flows.requires_grad_(True)

    # -- exec fold fxns --
    search = stnls.search.PairedSearch(ws, ps, K, nheads,
                                       dilation=dilation,
                                       stride0=stride0, stride1=stride1,
                                       reflect_bounds=reflect_bounds,full_ws=True,
                                       self_action=self_action,
                                       dist_type=dist_type,itype=itype)
    refine = stnls.search.PairedRefine(ws, wr, K, kr, ps, nheads,
                                       dilation=dilation,
                                       stride0=stride0, stride1=stride1,
                                       reflect_bounds=reflect_bounds,full_ws=True,
                                       self_action=self_action,
                                       dist_type=dist_type,itype=itype)

    # -- test api --
    dists_gt,inds_gt = search.paired_vids(vid0, vid1, flows, wt)
    dists_te,inds_te = refine.paired_vids(vid0, vid1, inds_gt, wt)

    # -- viz --
    # print(th.cat([inds_gt,inds_te],-1))

    # -- compare --
    assert th.allclose(dists_te,dists_gt,1e-3,1e-3,equal_nan=True)

def test_fwd_anchor(ws,wr,ps,pt,stride0,stride1,dilation,
                    nheads,dist_type,itype,seed,reflect_bounds):

    # -- get args --
    set_seed(seed)
    device = "cuda:0"
    Ks = ws*ws
    K = ws*ws*wr*wr
    kr = 1.
    self_action = "anchor"

    # -- load data --
    B,HD,F,H,W = 1,nheads,3,16,16
    frame = th.ones((B,F,H,W),device=device).float()
    frame0 = th.randn_like(frame)-0.5
    frame1 = th.randn_like(frame)

    # -- load flows --
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    flows = th.ones((B,HD,nH,nW,Ks,2)).to(device)+0.1
    flows = th.rand_like(flows)/2.+th.randint_like(flows,-2,2)+0.2
    # flows = th.zeros_like(flows)

    # -- exec fold fxns --
    k0 = wr*wr
    search0 = stnls.search.PairedRefine(ws, wr, -1, kr, ps, nheads, dilation=dilation,
                                        stride0=stride0, stride1=stride1,
                                        reflect_bounds=reflect_bounds,full_ws=False,
                                        self_action=None, dist_type=dist_type,
                                        itype=itype)
    k1 = 3
    search1 = stnls.search.PairedRefine(ws, wr, k1, kr, ps, nheads, dilation=dilation,
                                        stride0=stride0, stride1=stride1,
                                        reflect_bounds=reflect_bounds,full_ws=False,
                                        self_action="anchor_each",dist_type=dist_type,
                                        topk_mode="each",itype=itype)
    k2 = 5
    search2 = stnls.search.PairedRefine(ws, wr, k2, kr, ps, nheads, dilation=dilation,
                                        stride0=stride0, stride1=stride1,
                                        reflect_bounds=reflect_bounds,full_ws=True,
                                        self_action="anchor_each", dist_type=dist_type,
                                        topk_mode="each",itype=itype)
    k3 = 8
    search3 = stnls.search.PairedRefine(ws, wr, k3, kr, ps, nheads, dilation=dilation,
                                        stride0=stride0, stride1=stride1,
                                        reflect_bounds=reflect_bounds,full_ws=True,
                                        self_action="anchor",dist_type=dist_type,
                                        topk_mode="each",itype=itype)


    # -- exec --
    HD = nheads
    vshape = (B,HD,nH,nW,Ks)

    dists0,inds0 = search0(frame0,frame1,flows)
    dists0,inds0 = dists0.view(vshape+(k0,)),inds0.view(vshape+(k0,2,))
    dists0 = dists0[...,:,wr//2*wr+wr//2]
    inds0= inds0[...,:,wr//2*wr+wr//2,:]

    dists1,inds1 = search1(frame0,frame1,flows)
    dists1,inds1 = dists1.view(vshape+(k1,)),inds1.view(vshape+(k1,2,))
    dists1 = dists1[...,:,0]
    inds1= inds1[...,:,0,:]

    dists2,inds2 = search2(frame0,frame1,flows)
    dists2,inds2 = dists2.view(vshape+(k2,)),inds2.view(vshape+(k2,2,))
    dists2 = dists2[...,:,0]
    inds2= inds2[...,:,0,:]

    dists3,inds3 = search3(frame0,frame1,flows)
    dists3,inds3 = dists3.view(vshape+(k3,)),inds3.view(vshape+(k3,2,))
    dists3 = dists3[...,:,0]
    inds3= inds3[...,:,0,:]

    # print(th.stack([dists0[...,0],dists1[...,0],dists2[...,0],dists3[...,0]],-1))
    # print(th.stack([inds0[...,0,:],inds1[...,0,:],inds2[...,0,:],inds3[...,0,:]],-2))


    # -- check all pairwise --
    dists = [dists0,dists1,dists2]
    inds = [inds0,inds1,inds2]
    for i in range(3):
        for j in range(3):
            if i == j: continue
            assert th.allclose(dists[i],dists[j],1e-3,1e-3,equal_nan=True)
            assert th.allclose(inds[i],inds[j],1e-3,1e-3,equal_nan=True)

    # -- check all against "anchor" --
    for i in range(3):
        assert th.allclose(dists[i][...,0],dists3[...,0],1e-3,1e-3,equal_nan=True)
        assert th.allclose(inds[i][...,0,:],inds3[...,0,:],1e-3,1e-3,equal_nan=True)

    # -- check against flow --
    def reflect_bounds(flow,i,L):
        args0 = th.where(flow[:,i] > (L-1))
        args1 = th.where(flow[:,i] < 0)
        flow[:,i][args0] = 2*(L-1) - flow[:,i][args0]
        flow[:,i][args1] = -flow[:,i][args1]
    grid = stnls.nn.index_grid(nH,nW).flip(1)*stride0
    for i in range(3):
        inds_i = inds[i]
        for ki in range(Ks):

            # -- unpack --
            ind = 1.*inds_i[...,ki,:]
            flow = rearrange(flows[...,ki,:],'b hd h w two -> (b hd) two h w')
            flow = flow.clone() + grid
            if itype == "int": flow = flow.round()

            # -- reflect --
            reflect_bounds(flow,0,H)
            reflect_bounds(flow,1,W)

            # -- normalize --
            flow = flow - grid

            # -- shaping --
            flow = rearrange(flow,'b i h w -> b h w i')

            # -- compare --
            diff = th.mean(th.abs(ind - flow)).item()
            assert th.allclose(ind,flow,1e-3,1e-3,equal_nan=True)

# @pytest.mark.slow
def test_refine_noshuffle_bwd(ws,wt,wr,kr,ps,pt,stride0,stride1,dilation,
                              self_action,K,nheads,dist_type,itype,seed,reflect_bounds):
    """

    Test the CUDA code with torch code

    Forward Pass

    """


    # -- init vars --
    device = "cuda:0"
    full_ws = True
    set_seed(seed)

    # -- shapes --
    W_t = 2*wt+1
    K,kr = W_t*ws*ws,-1
    HD = nheads

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
    vid0.requires_grad_(True)
    vid1.requires_grad_(True)

    # -- exec fold fxns --
    wr = 1
    refine = stnls.search.PairedRefine(ws, wr, -1, kr, ps, nheads,
                                       dilation=dilation,
                                       stride0=stride0, stride1=stride1,
                                       reflect_bounds=reflect_bounds,full_ws=full_ws,
                                       self_action=self_action,
                                       dist_type=dist_type,itype=itype,topk_mode="all")

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
    ref_dists,ref_inds = refine.paired_vids(vid0,vid1,srch_inds,wt)

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



# def test_anchor_fwd(ws,wt,wr,ps,stride0,stride1,dilation,
#                     nheads,dist_type,itype,seed,reflect_bounds):

#     """

#     Test the CUDA code with torch code

#     Forward Pass

#     """

#     # -- init vars --
#     dil = dilation
#     pt = 1
#     device = "cuda:0"
#     clean_flow = True
#     comp_flow = False
#     use_adj = False
#     full_ws = False
#     ext = "jpg"
#     dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
#     topk_mode = "each"
#     kr = -1
#     set_seed(seed)

#     # -- load data --
#     vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
#     vid = vid.to(device)[:,:5,:3,::2,::2].contiguous()
#     vid = repeat(vid,'b t c h w -> b t (r c) h w',r=12)[:,:32].contiguous()
#     vid0 = th.rand_like(vid)-0.5
#     vid1 = th.rand_like(vid)-0.5

#     # -- compute flow --
#     B,T,F,H,W = vid.shape
#     W_t = 2*wt+1
#     nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
#     flows = 2*th.rand((B,T,W_t-1,2,nH,nW)).to(vid0.device)

#     # -- exec fold fxns --
#     k0 = 5
#     search0 = stnls.search.NonLocalSearch(ws, wt, ps, k0, nheads,
#                                           dilation=dil,stride0=stride0, stride1=stride1,
#                                           reflect_bounds=reflect_bounds,full_ws=True,
#                                           self_action=None,use_adj=use_adj,
#                                           dist_type=dist_type,topk_mode="each",
#                                           itype=itype)
#     k = 1
#     refine0 = stnls.search.RefineSearch(ws, wt, wr, k, kr, ps, nheads,
#                                         dilation=dil,stride0=stride0, stride1=stride1,
#                                         reflect_bounds=reflect_bounds,full_ws=False,
#                                         self_action="anchor_each",use_adj=use_adj,
#                                         dist_type=dist_type,topk_mode="each",itype=itype)
#     k = 1
#     refine1 = stnls.search.RefineSearch(ws, wt, wr, k, kr, ps, nheads,
#                                         dilation=dil,stride0=stride0, stride1=stride1,
#                                         reflect_bounds=reflect_bounds,full_ws=True,
#                                         self_action="anchor_each",use_adj=use_adj,
#                                         dist_type=dist_type,topk_mode="each",itype=itype)


#     # -- exec search --
#     HD = nheads
#     vshape = (B,HD,T,nH,nW,W_t*k0)
#     dists0,inds0 = search0(vid0,vid1,flows)
#     dists0,inds0 = dists0.view(vshape),inds0.view(vshape+(3,))

#     # -- exec refine --
#     dists_r0,inds_r0 = refine0(vid0,vid1,inds0)
#     dists_r1,inds_r1 = refine1(vid0,vid1,inds0)
#     # print(th.stack([dists0,dists_r0],-1))
#     # print(th.stack([inds0,inds_r0],-1))
#     # print(th.stack([dists0,dists_r1],-1))
#     # args = th.where(th.abs(dists0-dists_r0)>1e-3)
#     # print(dists0[args][:10])
#     # print(dists_r0[args][:10])

#     # -- compare --
#     assert th.allclose(dists0,dists_r0,1e-2,1e-3,equal_nan=True)
#     assert th.allclose(inds0,inds_r0,1e-3,1e-3,equal_nan=True)
#     assert th.allclose(dists0,dists_r1,1e-3,1e-3,equal_nan=True)
#     assert th.allclose(inds0,inds_r1,1e-3,1e-3,equal_nan=True)


# def test_fwd_topk(ws,wt,wr,ps,stride0,stride1,dilation,dist_type,seed,reflect_bounds):

#     """

#     Test the CUDA code with torch code

#     Forward Pass

#     """

#     # -- init vars --
#     dil = dilation
#     pt = 1
#     device = "cuda:0"
#     clean_flow = True
#     comp_flow = False
#     use_adj = False
#     full_ws = True
#     ext = "jpg"
#     dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
#     topk_mode = "each"
#     itype = "float"
#     nheads = 1
#     kr = -1
#     set_seed(seed)

#     # -- load data --
#     vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
#     vid = vid.to(device)[:,:5,:3,:,:].contiguous()
#     vid = repeat(vid,'b t c h w -> b t (r c) h w',r=12)[:,:32].contiguous()
#     vid0 = th.rand_like(vid)-0.5
#     vid1 = th.rand_like(vid)-0.5

#     # -- compute flow --
#     B,T,F,H,W = vid.shape
#     W_t = 2*wt+1
#     nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
#     flows = 2*th.rand((B,T,W_t-1,2,nH,nW)).to(vid0.device)

#     # -- exec fold fxns --
#     k0 = W_t*ws*ws
#     search = stnls.search.NonLocalSearch(ws, wt, ps, -1, nheads,
#                                          dilation=dil,stride0=stride0, stride1=stride1,
#                                          reflect_bounds=reflect_bounds,full_ws=True,
#                                          self_action=None,use_adj=use_adj,
#                                          dist_type=dist_type,topk_mode="all",
#                                          itype=itype)
#     k = k0*wr*wr
#     stride1 = 0.1
#     refine = stnls.search.RefineSearch(ws, wt, wr, k, kr, ps, nheads,
#                                        dilation=dil,stride0=stride0, stride1=stride1,
#                                        reflect_bounds=reflect_bounds,full_ws=True,
#                                        self_action=None,use_adj=use_adj,
#                                        dist_type=dist_type,topk_mode="all",itype=itype)

#     # -- exec --
#     _dists,_inds = search(vid0,vid1,flows)
#     dists,inds = refine(vid0,vid1,_inds)
#     # print(_inds[0,0,0,2,2,:10])
#     # print(inds[0,0,0,2,2,:10])

#     delta = dists[...,1:] - dists[...,:-1]
#     delta = delta[~th.isnan(delta)]
#     if dist_type == "l2":
#         assert th.all(delta>=0).item()
#     else:
#         assert th.all(delta<=0).item()

