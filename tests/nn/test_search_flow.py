# -- data mgnmt --
from pathlib import Path
from dev_basics.utils import vid_io

# -- testing --
import pytest
import random

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- patchify --
from torch.nn.functional import fold,unfold,pad

# -- stnls --
import stnls
import stnls.utils.gpu_mem as gpu_mem
from stnls.utils.pads import comp_pads
from stnls.utils.inds import get_batching_info

# -- paths --
SAVE_DIR = Path("./output/tests/prod_search")

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def pytest_generate_tests(metafunc):
    test_lists = {"b":[1],"t":[2],"h":[128],"w":[128],"seed":[123],
                  "wt":[4],"stride0":[1]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def run_compare(tensor_gt,tensor_te,mean_tol,max_tol,small_tol=1e-3):

    # -- compute diffs --
    cond_a = th.logical_not(th.isinf(tensor_gt))
    cond_b = tensor_gt.abs() > small_tol
    args0 = th.where(th.logical_and(cond_a,cond_b)) # remove all inf
    diff = th.abs(tensor_te - tensor_gt) / (tensor_gt.abs()+1e-4)
    # print("[a] num diff: ",diff.numel())
    diff = diff[args0]
    # print("[b] num diff: ",diff.numel())
    N = diff.numel()

    # -- viz --
    args1 = th.where(diff.abs() > 1e-3)
    if len(tensor_gt[args0][args1]) < max(20,int(N/100.)): # allow a few to be different
        diff = diff[th.where(diff.abs() < 1e-3)]
    # print(len(tensor_gt[args0][args1]))
    # print(tensor_gt[args0][args1])
    # print(tensor_te[args0][args1])
    # if len(tensor_gt[args0][args1]) > 0:
    #     print(tensor_gt[args0][args1][0].item())
    #     print(tensor_te[args0][args1][0].item())
    # print(th.where(th.abs(tensor_gt[0] - tensor_te[0]) > 1e-1))

    # -- test --
    error = diff.mean().item()
    if error > mean_tol: print("error: ",error)
    assert error < mean_tol

    max_error = diff.max().item()
    if max_error > max_tol: print("max error: ",max_error)
    assert max_error < max_tol

# def extract_search_from_accumulated(aflows,wt,stride0):

#     # -- setup --
#     T = aflows.fflow.shape[1]
#     W_t = 2*wt+1
#     flows = []

#     for ti in range(T):
#         # -- bounds for ti --
#         t_shift = min(0,ti - wt) + max(0,ti + wt - (T-1))
#         t_max = min(T-1,ti + wt - t_shift)
#         flows_t = []
#         for si in range(1,W_t):
#             # -- select adjacent frame --
#             tj = ti + si
#             tj = t_max - si if (tj > t_max) else tj
#             dt = tj - ti
#             print(ti,tj,t_max)
#             flow_gt = aflows.fflow[:,ti,dt-1] if (ti < tj) else aflows.bflow[:,ti,-dt-1]
#             flows_t.append(flow_gt[...,::stride0,::stride0])
#         flows_t = th.stack(flows_t,1)
#         flows.append(flows_t)
#     flows = th.stack(flows,1)
#     return flows

def test_fwd(seed,b,t,h,w,wt,stride0):
    """

    Test the CUDA code with torch code

    Forward Pass

    """


    # -- config --
    device = "cuda:0"
    set_seed(seed)
    zflow = th.zeros((1,t,2,h,w),device=device,dtype=th.float32)

    # -- compute flow --
    M = 1.1
    flows = edict()
    flows.fflow = th.round(th.randn_like(zflow)*3,decimals=5)
    flows.fflow = th.round(th.randn_like(zflow)/2.,decimals=8)
    flows.bflow = th.round(th.randn_like(zflow)/2.,decimals=8)
    M = 2
    flows.fflow = th.clamp(flows.fflow,-M,M).round()
    flows.bflow = th.clamp(flows.bflow,-M,M).round()
    # print(flows.fflow.shape,flows.bflow.shape)

    # -- init data --
    fflow_gt = flows.fflow.clone().requires_grad_(True)
    bflow_gt = flows.bflow.clone().requires_grad_(True)
    fflow_te = flows.fflow.clone().requires_grad_(True)
    bflow_te = flows.bflow.clone().requires_grad_(True)
    aflows_gt = stnls.nn.accumulate_flow(fflow_gt,bflow_gt)
    extract = stnls.nn.extract_search_from_accumulated
    # print(aflows_gt.fflow.shape)
    flows_gt = extract(aflows_gt.fflow,aflows_gt.bflow,wt,stride0)
    flows_te = stnls.nn.search_flow(fflow_te,bflow_te,wt,stride0)

    # -- comparison info --
    mean_tol = 5e-3
    max_tol = 1e-3
    sm_tol = 1e-2
    run_compare(flows_gt,flows_te,mean_tol,max_tol,sm_tol)
    run_compare(flows_gt[0,:t-1,0],fflow_gt[0,:t-1],
                mean_tol,max_tol,sm_tol)

def test_bwd(seed,b,t,h,w,wt,stride0):
    """

    Test the CUDA code with torch code

    Forward Pass

    """


    # -- config --
    device = "cuda:0"
    set_seed(seed)
    zflow = th.zeros((b,t,2,h,w),device=device,dtype=th.float32)

    # -- compute flow --
    # M = 2.1
    M = 2.1
    # fflow = th.ones_like(zflow)
    # bflow = th.ones_like(zflow)
    # fflow[0,1,0,1,1] = .1
    # fflow = th.round(th.clamp(th.randn_like(zflow),1e-3,1e-2),decimals=4)
    # bflow = th.round(th.clamp(th.randn_like(zflow),1e-3,1e-2),decimals=4)
    fflow = th.round(th.clamp(th.rand_like(zflow)*1e-2,1e-4,1e-2),decimals=6)
    bflow = th.round(th.clamp(th.rand_like(zflow)*1e-2,1e-4,1e-2),decimals=6)

    # fflow = -th.round(th.clamp(th.randn_like(zflow),1e-3,1e-2),decimals=4)
    # bflow = -th.round(th.clamp(th.randn_like(zflow),1e-3,1e-2),decimals=4)

    # # fflow = th.ones_like(fflow)
    # # fflow[...,:5,:5] = .1
    # # fflow[:,0] = th.ones_like(fflow[:,1])*.99999
    # fflow[...] = 0
    # fflow[:,0] = th.ones_like(fflow[:,1])*1.
    # fflow[:,1] = th.clamp(th.randn_like(fflow[:,1]),0,M)
    # fflow[:,1] = th.clamp(th.randn_like(fflow[:,1]),0,1).round()/2.+0.25
    # fflow[:,1] = th.ones_like(fflow[:,1])*0.2
    # fflow[:,1,:,3:,3:] = 0.3
    # fflow[:,2:] = 0
    # fflow[:,0,:,0,0] = 2
    # fflow[:,0,:,1,1] = 3
    # bflow[...] = 0
    # fflow = th.round(th.randn_like(zflow),decimals=5).round()
    # bflow = th.round(th.randn_like(zflow),decimals=5).round()

    # -- enable grads --
    fflow_gt = fflow.clone().requires_grad_(True)
    bflow_gt = bflow.clone().requires_grad_(True)
    fflow_te = fflow.clone().requires_grad_(True)
    bflow_te = bflow.clone().requires_grad_(True)

    # -- accumulate --
    aflows_gt = stnls.nn.accumulate_flow(fflow_gt,bflow_gt,fwd_mode="pytorch")
    extract = stnls.nn.extract_search_from_accumulated
    flows_gt = extract(aflows_gt.fflow,aflows_gt.bflow,wt,stride0)
    # aflows_gt = stnls.nn.accumulate_flow(fflow_gt,bflow_gt,fwd_mode="pytorch")
    flows_te = stnls.nn.search_flow(fflow_te,bflow_te,wt,stride0)

    # def fxn_gt(fflow_gt):
    #     aflows_gt = stnls.nn.accumulate_flow(fflow_gt,bflow_gt,fwd_mode="pytorch")
    #     extract = stnls.nn.extract_search_from_accumulated
    #     flows_gt = extract(aflows_gt.fflow,aflows_gt.bflow,wt,stride0)
    #     return flows_gt
    # print(th.autograd.gradcheck(fxn_gt,(fflow_gt)))

    # print(flows_te.shape)
    # print("="*30)
    # print(flows_te[0,0,0])
    # print("="*30)
    # print(flows_te[0,0,1])
    # print(th.mean((flows_te[0,0,1] - fflow[0,1])**2))
    # print("="*30)
    # print(flows_te[0,0,2])
    # print("="*30)

    # -- run autograd --
    grad = th.randn_like(flows_gt)
    # grad[:,1:,0:] = 0
    # grad[:,0,:] = 0
    # grad[:,:,0] = 0
    # grad[:,:,2:] = 0
    # # grad[:,0,:] = 0
    # grad[:,1:,:] = 0
    # grad[...] = 0
    # grad[:,0,1] = th.randn_like(grad[:,0,1])
    # grad[:,0,1] = th.ones_like(grad[:,0,1])
    # grad[:,0,0,0,:4,:4] = th.ones_like(grad[:,0,0,0,:4,:4])
    # grad[:,0,1,0,:4,:4] = th.ones_like(grad[:,0,1,0,:4,:4])
    # print("grad.shape: ",grad.shape)
    th.autograd.backward(flows_gt,grad,retain_graph=True)
    th.autograd.backward(flows_te,grad,retain_graph=True)

    # -- compare --
    mean_tol = 5e-3
    max_tol = 1e-2
    sm_tol = 1e-3
    grads_gt = [fflow_gt.grad,bflow_gt.grad]
    grads_te = [fflow_te.grad,bflow_te.grad]
    for grad_gt,grad_te in zip(grads_gt,grads_te):

        # grad_gt = grad_gt.round(decimals=2)
        # grad_te = grad_te.round(decimals=2)

        # print("-="*20)
        # print(grad_gt.shape)
        # print(grad_gt[0,0,:,:5,:5])
        # print(grad_te[0,0,:,:5,:5])
        # print("-"*20)

        # print(grad_gt[0,0,:,5:10,5:10])
        # print(grad_te[0,0,:,5:10,5:10])

        # print("-"*20)

        # print(grad_gt[0,0,:,5:7,5:7])
        # print(grad_te[0,0,:,5:7,5:7])

        # print("-"*20)
        # print("-"*20)

        # print(grad_gt[0,1,:,:4,:4])
        # print(grad_te[0,1,:,:4,:4])
        # print("-"*20)
        # print(grad_gt[0,1,:,5:10,5:10])
        # print(grad_te[0,1,:,5:10,5:10])
        # print("-"*20)
        # print("-"*20)

        # print(grad_gt[0,2,:,:4,:4])
        # print(grad_te[0,2,:,:4,:4])
        # print("-"*20)


        # print(grad_gt[0,5,:,:4,:4])
        # print(grad_te[0,5,:,:4,:4])

        # print(grad_gt[0,6,:,:4,:4])
        # print(grad_te[0,6,:,:4,:4])

        # -- viz --
        # print("grad_gt.shape: ",grad_gt.shape)
        gdiff = th.abs(grad_gt[:,:] - grad_te[:,:]).mean(-3,keepdim=True)
        gdiff /= gdiff.max()
        # print(gdiff.shape)
        vid_io.save_video(gdiff,'output/tests/nn/search_flow/','gdiff')

        # -- run test --
        run_compare(grad_gt,grad_te,mean_tol,max_tol,sm_tol)

    # -- dev --
    # afflow_gt = acc_flows_gt.fflow
    # afflow_te = acc_flows_te.fflow
    # extra_info(afflow_gt,afflow_te,fflow_gt,fflow_te,"fwd")

    # abflow_gt = acc_flows_gt.bflow
    # abflow_te = acc_flows_te.bflow
    # extra_info(abflow_gt,abflow_te,bflow_gt,bflow_te,"bwd")


def extra_info(aflow_gt,aflow_te,flow_gt,flow_te,mode):

    # return
    print("-="*10 + " gt " + "-="*10)
    if mode == "fwd":
        print(flow_gt.grad[0,:3,:,:5,:5].round(decimals=3))
    else:
        print(flow_gt.grad[0,-3:,:,:5,:5].round(decimals=3))
    print("-="*10 + " te " + "-="*10)
    if mode == "fwd":
        print(flow_te.grad[0,:3,:,:5,:5].round(decimals=3))
    else:
        print(flow_te.grad[0,-3:,:,:5,:5].round(decimals=3))
    print("-="*10 + " ratio " + "-="*10)
    print((flow_gt.grad/flow_te.grad)[0].round(decimals=3))
    ratio = flow_gt.grad/flow_te.grad
    print(ratio[0])
    print(ratio[0,0])
    # print(ratio[0,0][th.where(th.abs(ratio[0,1]-1.3333)>1e-2)])
    print(th.mean((flow_gt.grad - flow_te.grad)**2).item())
    print(th.mean((flow_gt.grad - flow_te.grad)**2).item())
    print(th.mean((flow_gt.grad[:,1:] - flow_te.grad[:,1:])**2).item())

    grad_gt =flow_gt.grad
    grad_te =flow_te.grad
    diff = th.abs(grad_gt - grad_te)/(grad_gt.abs()+1e-3)
    args = th.where(diff > 1e-2)
    print(args)
    print(flow_te[args])
    print(len(args[0]))
    print(th.any(th.abs(th.round(aflow_te) - aflow_te)<1e-9).item())
    # for t in range(4):
    #     isint = th.abs(th.round(aflow_te[0,t,:4-t]) - aflow_te[0,t,:4-t])<1e-9
    #     print(th.sum(1*isint))
    print(th.stack([grad_gt[args],grad_te[args]],-1))

    # -- compare fwd --
    diff = th.abs(aflow_gt - aflow_te)/(aflow_gt.abs()+1e-3)
    args = th.where(diff > 1e-2)
    print(args)
    print(th.mean((aflow_gt - aflow_te)**2))
    print(th.stack([aflow_gt[args],aflow_te[args]],-1)[:10])
