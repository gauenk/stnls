
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

# -- paths --
SAVE_DIR = Path("./output/tests/graph_opts/scatter")

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def pytest_generate_tests(metafunc):
    test_lists = {"nheads":[1],"ws":[9],"wt":[1],
                  "stride0":[1],"stride1":[1],
                  "full_ws":[True],"seed":[123]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)


def test_scatter_labels(nheads,ws,wt,stride0,stride1,full_ws,seed):
    """

    Test the CUDA code with torch code

    Forward Pass

    """

    # -- get args --
    device = "cuda:0"
    set_seed(seed)
    W_t = 2*wt+1
    B,HD,T,F,H,W = 1,nheads,3,3,32,32
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    K = ws*ws*W_t
    itype = "int"

    # -- load flows --
    flows = th.ones((B,HD,T,W_t-1,2,nH,nW)).cuda()/2.
    flows = th.rand_like(flows)/2.+th.randint_like(flows,-3,3)+0.2
    flows = th.zeros_like(flows)
    flows = flows.round().int()

    # -- load top-k flows --
    ps = 1
    search = stnls.search.NonLocalSearch(ws, wt, ps, K, nheads, dist_type="l2",
                                         stride0=stride0, stride1=stride1,
                                         reflect_bounds=True,self_action="anchor",
                                         itype="int",full_ws=full_ws)
    vid = th.rand((B,HD,T,F,H,W)).to(device)
    dists,flows_k = search(vid,vid,flows)
    # flows_k[ibatch][ihead][ti][hi][wi][ki][_idx]
    # print(flows_k[0,0,0,:2,:2])

    # -- get unique labels --
    names,labels = stnls.graph_opts.scatter_labels(flows,flows_k,ws,wt,stride0,stride1,full_ws)

    if (full_ws):
        Q = T*H*W
        assert(int(th.sum(labels>=0).item())==Q*K)
    else:
        print("no test for scatter_labels.")

def test_scatter_tensor(nheads,ws,wt,stride0,stride1,full_ws,seed):


    # -- get args --
    device = "cuda:0"
    set_seed(seed)
    W_t = 2*wt+1
    B,HD,T,F,H,W = 1,nheads,3,3,32,32
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    K = ws*ws*W_t
    itype = "int"
    K = 30

    # -- load flows --
    flows = th.ones((B,HD,T,W_t-1,2,nH,nW)).cuda()/2.
    flows = th.rand_like(flows)/2.+th.randint_like(flows,-3,3)+0.2
    flows = th.zeros_like(flows)
    flows = flows.round().int()

    # -- load top-k flows --
    ps = 1
    search = stnls.search.NonLocalSearch(ws, wt, ps, K, nheads, dist_type="l2",
                                         stride0=stride0, stride1=stride1,
                                         reflect_bounds=True,self_action="anchor",
                                         itype="int",full_ws=full_ws)
    vid = th.rand((B,HD,T,F,H,W)).to(device)
    dists,flows_k = search(vid,vid,flows)
    # flows_k[ibatch][ihead][ti][hi][wi][ki][_idx]
    # print(flows_k[0,0,0,:2,:2])

    # -- get unique labels --
    names,labels = stnls.graph_opts.scatter_labels(flows,flows_k,ws,wt,stride0,stride1,full_ws)
    print(labels)
    print(labels.min(),labels.max())

    # -- weighted --
    gather_weights = th.softmax(-dists,-1)
    scatter_weights = stnls.graph_opts.scatter_tensor(gather_weights,flows_k,labels,stride0)
    Q = T*H*W
    print(scatter_weights.shape)
    print(int(th.sum(scatter_weights>=0).item()),Q*K)

    scatter_flows_k = stnls.graph_opts.scatter_tensor(flows_k,flows_k,labels,stride0)
    scatter_flows_k = -scatter_flows_k
    # scatter_flows_k[th.where(scatter_flows_k==th.inf)] = -th.inf

    # K = 30
    K0 = K//2
    s_weight,s_flows_k = stnls.graph_opts.scatter_topk(scatter_weights,scatter_flows_k,K0)
    s_weight = rearrange(s_weight,'b hd (t h w) k -> b hd t h w k',t=T,h=H)
    print(s_weight[0,0,0,:3,:3])
    print(s_weight[0,0,0,10:12,10:12])
    s_weight = th.softmax(s_weight,-1)
    print("-"*30)
    print(s_weight[0,0,0,:3,:3])
    print(s_weight[0,0,0,10:12,10:12])
    # print(s_weight[0,0,0,:3,:3])
    s_flows_k = rearrange(s_flows_k,'b hd (t h w) k tr -> b hd t h w k tr',t=T,h=H)
    # print(s_weight.shape)
    # print(int(th.sum(s_weight>=0).item()),Q*K)
    # return
    return

    agg = stnls.agg.NonLocalGather(ps=ps,stride0=stride0,itype="int")
    print(vid.shape,s_weight.shape,s_flows_k.shape)
    stack = agg(vid,s_weight,s_flows_k)
    print(stack[0,0])
    print(stack.shape)
    # print(w[0,0,-1])
    # print(f[0,0,-1])

    # print(flows_k[0,0,-1,-1,-1])
    # print(flows_k[0,0,-1,-1-3,-1-7])

    if (full_ws):
        Q = T*H*W
        print(int(th.sum(scatter_weights>=0).item()),Q*K)
        # assert(int(th.sum(scatter_weights>=0).item())==Q*K)
    else:
        print("no test for scatter_weigths.")

    # # -- scatter patches --
    # scatter,mask = stnls.agg.NonLocalScatter(ps,stride0)
    # stack = scatter(vid,scatter_weights,flows_k,labels)
    # print(stack.shape)

# def test_fwd(ps,stride0,K,nheads,reflect_bounds,itype,seed):

#     """

#     Test the CUDA code with torch code

#     Forward Pass

#     """

#     # -- get args --
#     pt = 1
#     device = "cuda:0"
#     set_seed(seed)

#     # -- load data --
#     B,HD,T,F,H,W = 1,nheads,2,1,8,8
#     vid = th.rand((B,T,F*HD,H,W),device=device).float()

#     # -- load weights --
#     nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
#     weights = th.rand((B,HD,T,nH,nW,K)).to(device)
#     # weights = th.ones_like(weights)
#     # vid = th.ones_like(vid)

#     # -- load flows --
#     flows = th.ones((B,HD,T,nH,nW,K,3))+0.1
#     flows = th.rand_like(flows)/2.+1.1
#     tgrid = th.arange(0,T).view(1,1,T,1,1,1)
#     flows[...,0] = th.randint(0,T,size=flows[...,0].shape)-tgrid
#     flows[...,1:] = th.rand_like(flows[...,1:])/2.+0.2
#     not_int = th.all(th.abs(flows[...,1:].round() - flows[...,1:])>1e-5).item()
#     if itype == "float":
#         assert not_int,"Gradcheck only works _not_ near an int."
#     else:
#         flows = flows.round().int()
#     flows = flows.to(vid.device)

#     # -- exec fold fxns --
#     agg = stnls.agg.NonLocalScatter(ps=ps,stride0=stride0,
#                                     reflect_bounds=reflect_bounds,
#                                     itype=itype)
#     stack = agg(vid,weights,flows)
#     # stack_gt = stnls.testing.non_local_scatter(vid,weights,flows,ps,stride0,
#     #                                           reflect_bounds=reflect_bounds,itype=itype)
#     # assert th.allclose(stack,stack_gt,1e-2,1e-2,equal_nan=True)

# def test_bwd(ps,stride0,K,nheads,reflect_bounds,itype,seed):

#     """

#     Test the CUDA code with torch code

#     Forward Pass

#     """

#     # -- get args --
#     pt = 1
#     device = "cuda:0"
#     set_seed(seed)

#     # -- load data --
#     B,HD,T,F,H,W = 1,nheads,2,3,8,8
#     vid = th.rand((B,T,F*HD,H,W),device=device).float().requires_grad_(True)

#     # -- load weights --
#     nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
#     weights = th.rand((B,HD,T,nH,nW,K)).to(device).requires_grad_(True)

#     # -- load flows --
#     flows = th.ones((B,HD,T,nH,nW,K,3))+0.1
#     flows = th.rand_like(flows)/2.+1.1
#     tgrid = th.arange(0,T).view(1,1,T,1,1,1)
#     flows[...,0] = th.randint(0,T,size=flows[...,0].shape)-tgrid
#     flows[...,1:] = th.rand_like(flows[...,1:])/2.+2.2
#     not_int = th.all(th.abs(flows[...,1:].round() - flows[...,1:])>1e-5).item()
#     flows = flows.to(vid.device)
#     flows_t,flows_sp = flows[...,[0]],flows[...,1:]
#     if itype == "float":
#         assert not_int,"Gradcheck only works _not_ near an int."
#     else:
#         flows = flows.round().int()

#     # -- exec fold fxns --
#     stacking = stnls.agg.NonLocalScatter(ps=ps,stride0=stride0,
#                                          reflect_bounds=reflect_bounds,
#                                          itype=itype)
#     stack = stacking(vid,weights,flows)

#     # -- gradcheck --
#     stack_weights = lambda weights: stacking(vid,weights,flows)
#     th.autograd.gradcheck(stack_weights, weights, eps=1e-4,
#                           atol=1e-2, nondet_tol=1e-7, raise_exception=True)
#     stack_vid = lambda vid: stacking(vid,weights,flows)
#     th.autograd.gradcheck(stack_vid, vid, eps=1e-4,
#                           atol=1e-2, nondet_tol=1e-7, raise_exception=True)
#     if itype == "float":
#         flows_sp = flows_sp.requires_grad_(True)
#         def stack_flows(flows_sp):
#             flows = th.cat([flows_t,flows_sp],-1)
#             return stacking(vid,weights,flows)
#         th.autograd.gradcheck(stack_flows, flows_sp, eps=1e-2,
#                               atol=1e-2, nondet_tol=1e-7, raise_exception=True)


