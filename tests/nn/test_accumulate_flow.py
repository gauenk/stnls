# -- data mgnmt --
from pathlib import Path

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
    test_lists = {"b":[1],"t":[5],"h":[64],"w":[64],"seed":[123]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def run_compare(tensor_gt,tensor_te,mean_tol,max_tol,small_tol=1e-3):

    # -- compute diffs --
    cond_a = th.logical_not(th.isinf(tensor_gt))
    cond_b = tensor_gt.abs() > small_tol
    args0 = th.where(th.logical_and(cond_a,cond_b)) # remove all inf
    diff = th.abs(tensor_te - tensor_gt) / (tensor_gt.abs()+1e-4)
    diff = diff[args0]

    # -- viz --
    args1 = th.where(diff.abs() > 1e-3)
    if len(tensor_gt[args0][args1]) < 100: # allow a few to be different
        diff = diff[th.where(diff.abs() < 1e-3)]
    print(len(tensor_gt[args0][args1]))
    print(tensor_gt[args0][args1])
    print(tensor_te[args0][args1])
    if len(tensor_gt[args0][args1]) > 0:
        print(tensor_gt[args0][args1][0].item())
        print(tensor_te[args0][args1][0].item())


    # -- test --
    error = diff.mean().item()
    if error > mean_tol: print("error: ",error)
    assert error < mean_tol

    max_error = diff.max().item()
    if max_error > max_tol: print("max error: ",max_error)
    assert max_error < max_tol

def test_fwd(seed,b,t,h,w):
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
    flows.fflow = th.clamp(flows.fflow,-M,M)
    flows.bflow = th.clamp(flows.bflow,-M,M)

    # -- init data --
    fflow_gt = flows.fflow.clone().requires_grad_(True)
    bflow_gt = flows.bflow.clone().requires_grad_(True)
    fflow_te = flows.fflow.clone().requires_grad_(True)
    bflow_te = flows.bflow.clone().requires_grad_(True)
    acc_flows_gt = stnls.nn.accumulate_flow(fflow_gt,bflow_gt)
    afflow_gt = acc_flows_gt.fflow
    acc_flows_te = stnls.nn.accumulate_flow(fflow_te,bflow_te,fwd_mode="stnls")

    # -- compare --
    mean_tol = 5e-3
    max_tol = 1e-3
    sm_tol = 1e-2
    run_compare(acc_flows_te.fflow,acc_flows_gt.fflow,mean_tol,max_tol,sm_tol)
    run_compare(acc_flows_te.bflow,acc_flows_gt.bflow,mean_tol,max_tol,sm_tol)


def test_bwd(seed,b,t,h,w):
    """

    Test the CUDA code with torch code

    Forward Pass

    """


    # -- config --
    device = "cuda:0"
    set_seed(seed)
    zflow = th.zeros((b,t,2,h,w),device=device,dtype=th.float32)

    # -- compute flow --
    M = 2.1
    # flows.fflow = th.round(th.clamp(th.randn_like(flows.fflow)/2.,-1.1,1.1),decimals=5)
    # fflow = th.round(th.clamp(th.randn_like(zflow)/2.,-M,M),decimals=9)
    # bflow = th.round(th.clamp(th.randn_like(zflow)/2.,-M,M),decimals=9)
    fflow = th.ones_like(zflow)+8e-7
    bflow = th.ones_like(zflow)
    fflow[0,1,0,1,1] = .1
    fflow = th.round(th.clamp(th.randn_like(zflow),-M,M),decimals=9)
    bflow = th.round(th.clamp(th.randn_like(zflow),M,M),decimals=9)
    fflow = th.round(th.randn_like(zflow),decimals=5)
    bflow = th.round(th.randn_like(zflow),decimals=5)


    # -- enable grads --
    fflow_gt = fflow.clone().requires_grad_(True)
    bflow_gt = bflow.clone().requires_grad_(True)
    fflow_te = fflow.clone().requires_grad_(True)
    bflow_te = bflow.clone().requires_grad_(True)

    # -- accumulate --
    acc_flows_gt = stnls.nn.accumulate_flow(fflow_gt,bflow_gt,fwd_mode="pytorch")
    acc_flows_te = stnls.nn.accumulate_flow(fflow_te,bflow_te,fwd_mode="stnls")
    acc_gt = [acc_flows_gt.fflow,acc_flows_gt.bflow]
    acc_te = [acc_flows_te.fflow,acc_flows_te.bflow]

    # -- run autograd --
    for i in range(2):
        grad = th.randn_like(acc_gt[i])
        # grad[...] = 0
        # grad[0,0,1,:,:2,:2] = 1
        th.autograd.backward(acc_gt[i],grad,retain_graph=True)
        th.autograd.backward(acc_te[i],grad,retain_graph=True)

    # -- compare --
    mean_tol = 5e-3
    max_tol = 1e-2
    sm_tol = 1e-1
    grads_gt = [fflow_gt.grad,bflow_gt.grad]
    grads_te = [fflow_te.grad,bflow_te.grad]
    for grad_gt,grad_te in zip(grads_gt,grads_te):
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
