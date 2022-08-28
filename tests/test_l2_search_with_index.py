"""
Write "inds" check;
verify the correct meshgrid is included in the exhaustive search

see pacnet's test for details.


"""

# -- python --
import cv2,tqdm,copy,pytest
import numpy as np
import unittest
import tempfile
import sys
import shutil
from pathlib import Path
from easydict import EasyDict as edict

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- dnls --
import dnls
import dnls.utils.gpu_mem as gpu_mem
from dnls.utils.pads import comp_pads
from dnls.utils.inds import get_batching_info

# -- check if reordered --
from scipy import optimize
SAVE_DIR = Path("./output/tests/l2_search_with_index")

#
# -- meshgrid --
#

def pytest_generate_tests(metafunc):
    # test_lists = {"ps":[3],"stride":[1],"dilation":[1,2],
    #               "top":[3],"btm":[62],"left":[2],"right":[62]}
    # test_lists = {"ps":[4],"stride":[1,2],"dilation":[2],
    #               "top":[4],"btm":[64],"left":[1],"right":[61]}
    # test_lists = {"ps":[3,4,5,6,7,8],"stride":[1,2,3,4,5],"dilation":[1,2,3,4,5],
    #               "top":[3],"btm":[57],"left":[7],"right":[57]}
    # test_lists = {"ps":[3],"stride":[2],"dilation":[2],
    #               "top":[3],"btm":[57],"left":[7],"right":[57]}
    test_lists = {"ps":[7],"stride0":[2,4],"stride1":[4],"dilation":[1],"wt":[1,3],
                  "ws":[15,32],"k":[30],"exact":[True],"reflect_bounds":[True],
                  "seed":[123,234,0,1,2,3,4]}
    # test_lists = {"ps":[3,4,5,6,7,8],"stride":[1,2,3,4,5,8],"dilation":[1,2,3,4,5,8],
    #               "top":[1,11],"btm":[50,57],"left":[3,7],"right":[57,30]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)

#
# -- forward testing --
#

def test_cu_vs_th_fwd(ps,stride0,stride1,dilation,reflect_bounds,exact):
    """

    Test the CUDA code with torch code

    Forward Pass

    """
    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    pt = 1
    wt = 0
    ws = -1
    k = -1
    # stride0 = stride
    # stride1 = 1
    search_abs = ws == -1
    use_k = k>0
    exact = True

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    adj = True
    only_full = True

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device)[:1,].contiguous()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],-2)

    # -- normalize --
    vid /= vid.max()
    # vidr = th.ones_like(vid)
    vidr = th.rand_like(vid)

    # -- compute flow --
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)
    flows.fflow = 5*th.randn_like(flows.fflow)
    flows.bflow = 5*th.randn_like(flows.bflow)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape
    chnls = vid.shape[1]

    # -- unpack --
    _,_,n0,n1 = get_batching_info(vid.shape,stride0,stride1,ps,dil)
    n_h0,n_w0 = n0[0],n0[1]
    n_h1,n_w1 = n1[0],n1[1]

    # -- init --
    use_adj = True
    h0_off,w0_off,_,_ = comp_pads(vid.shape, ps, stride0, 1)
    h1_off,w1_off,_,_ = comp_pads(vid.shape, ps, stride1, 1)
    search = dnls.search.init("l2_with_index",
                              flows.fflow, flows.bflow, k, ps, pt,
                              ws, wt, dilation=dil,
                              stride0=stride0, stride1=stride1,
                              use_k = use_k,use_adj=use_adj,
                              reflect_bounds=reflect_bounds,
                              search_abs=search_abs,exact=exact,
                              h0_off=h0_off,w0_off=w0_off,
                              h1_off=h1_off,w1_off=w1_off)

    # -- batching info --
    n_h0,n_w0 = search.query_batch_info(vid.shape) # just showing api
    ntotal = t * n_h0 * n_w0
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- query inds --
    qindex = 0

    # -- run search --
    mode = "reflect" if reflect_bounds else "zero"
    score_gt = dnls.simple.search_nn.run_nn(vid,ps,stride=stride0,mode=mode,
                                            dilation=dil,vid1=vidr,stride1=stride1)
    if ws == -1:
        score_gt = rearrange(score_gt,'(sh sw) (h w) -> h w sh sw',sh=n_h0,h=n_h1)


    # -- testing code --
    score_te,inds_te = search(vid,qindex,ntotal,vid1=vidr)
    if ws == -1:
        score_te = rearrange(score_te,'(sh sw) (h w) -> h w sh sw',sh=n_h0,h=n_h1)

    # -- compare --
    tol = 1e-5
    error = th.mean(th.abs(score_te - score_gt)/score_gt.abs()).item()
    if error > tol: print("error: ",error)
    assert error < tol

    tol = 1e-4
    max_error = th.abs((score_te - score_gt)/score_gt.abs()).max().item()
    if max_error > tol: print("max error: ",max_error)
    assert max_error < tol

def test_cu_vs_simp_fwd(ws,wt,k,ps,stride0,stride1,dilation,reflect_bounds,exact,
                        seed):

    # -- set seed --
    set_seed(seed)

    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    pt = 1
    # wt = 3
    # ws = -1
    # stride0 = stride
    # stride1 = 1
    search_abs = k<=0
    use_k = k>0
    exact = True
    use_adj = True

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    adj = True
    only_full = True

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device)[:5].contiguous()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    # vid = th.cat([vid,vid],-1)
    vid = th.cat([vid,vid],-1)
    vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],-2)
    print("vid.shape: ",vid.shape)

    # -- normalize --
    vid /= vid.max()
    # vidr = th.ones_like(vid)
    vidr = th.rand_like(vid)
    vidr = th.rand_like(vid)
    print(ws,wt,k,ps,stride0,stride1)

    # -- compute flow --
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)
    flows.fflow = (10*th.randn_like(flows.fflow)).int().float()
    flows.bflow = (10*th.randn_like(flows.bflow)).int().float()
    # print(flows.fflow.abs().max())
    # print(flows.bflow.abs().max())
    # flows.fflow = 10*th.randn_like(flows.fflow).float()
    # flows.bflow = 10*th.randn_like(flows.bflow).float()


    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape
    chnls = vid.shape[1]

    # -- batching --
    _,_,n0,n1 = get_batching_info(vid.shape,stride0,stride1,ps,dil)
    n_h0,n_w0 = n0[0],n0[1]
    n_h1,n_w1 = n1[0],n1[1]

    # -- exec fold fxns --
    h0_off,w0_off,_,_ = comp_pads(vid.shape, ps, stride0, 1)
    h1_off,w1_off,_,_ = comp_pads(vid.shape, ps, stride1, 1)
    search = dnls.search.init("l2_with_index",
                              flows.fflow, flows.bflow, k, ps, pt,
                              ws, wt, dilation=dil,
                              stride0=stride0, stride1=stride1,
                              use_k = use_k,use_adj=use_adj,
                              reflect_bounds=reflect_bounds,
                              search_abs=search_abs,exact=exact,
                              h0_off=h0_off,w0_off=w0_off,
                              h1_off=h1_off,w1_off=w1_off)

    # -- batching info --
    n_h0,n_w0 = search.query_batch_info(vid.shape) # just showing api
    ntotal = t * n_h0 * n_w0
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- query inds --
    qindex = 0

    # -- run search --
    iqueries = dnls.utils.inds.get_query_batch(qindex,nbatch,stride0,t,h,w,device)
    score_gt,inds_gt = dnls.simple.search.run(vid,iqueries,flows,k,ps,pt,ws,wt,chnls,
                                              dilation=dil,stride=stride1,
                                              use_adj=use_adj,
                                              reflect_bounds=reflect_bounds,
                                              search_abs=search_abs,
                                              h0_off=h0_off,w0_off=w0_off,
                                              h1_off=h1_off,w1_off=w1_off,
                                              vid1=vidr)

    # -- testing code --
    score_te,inds_te = search(vid,qindex,ntotal,vid1=vidr)

    # -- viz --
    # print(ws)
    # print(score_te[0,:3])
    # print(score_gt[0,:3])
    # args = th.where(th.abs(score_te - score_gt) > 1e-3)
    # print(args)
    # print(score_te[args])
    # print(score_gt[args])
    # print("-"*10)
    # print(inds_te[63,:3])
    # print(inds_gt[63,:3])
    # print(inds_te[1055,:3])
    # print(inds_gt[1055,:3])
    # print("-"*10)
    # print(inds_te[...,0][args])
    # print(inds_te[...,1][args])
    # print(inds_te[...,2][args])
    # print("-"*10)
    # print(inds_gt[...,0][args])
    # print(inds_gt[...,1][args])
    # print(inds_gt[...,2][args])
    # print("-"*10)

    # -- compare --
    eps = 1e-5
    tol = 1e-5
    diff = th.abs(score_te - score_gt)/(score_gt.abs()+eps)
    args = th.where(score_gt.abs()>eps)

    error = diff.mean().item()
    if error > tol: print("error: ",error)
    assert error < tol

    tol = 1e-4
    diff = th.abs(score_te - score_gt)
    max_error = diff.max().item()
    if max_error > tol: print("max error: ",max_error)
    assert max_error < tol

def test_exact_bwd(ps,k,stride0,stride1,dilation,reflect_bounds):


    # -- get args --
    seed = 123
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    k,pt = 10,1
    wt = 3
    ws = -1
    # stride0 = stride
    # stride1 = 1
    search_abs = k<=0
    use_k = k>0
    if ws == -1 and k > 0: ws = 10
    exact = True
    use_adj = False
    use_rand = False

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    only_full = True

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device)[:3].contiguous()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],-2)

    # -- normalize --
    vid /= vid.max()
    # vidr = th.ones_like(vid)
    th.manual_seed(seed)
    vidr = th.rand_like(vid)

    # -- compute flow --
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape
    chnls = vid.shape[1]

    # -- batching --
    _,_,n0,n1 = get_batching_info(vid.shape,stride0,stride1,ps,dil)
    n_h0,n_w0 = n0[0],n0[1]
    n_h1,n_w1 = n1[0],n1[1]

    # -- exec fold fxns --
    h0_off,w0_off,_,_ = comp_pads(vid.shape, ps, stride0, 1)
    h1_off,w1_off,_,_ = comp_pads(vid.shape, ps, stride1, 1)
    h0_off,w0_off = 0,0
    h1_off,w1_off = 0,0
    search = dnls.search.init("l2_with_index",
                              flows.fflow, flows.bflow, k, ps, pt,
                              ws, wt, dilation=dil,
                              stride0=stride0, stride1=stride1,
                              use_k = use_k,use_adj=use_adj,
                              reflect_bounds=reflect_bounds,
                              search_abs=search_abs,exact=exact,
                              use_rand=use_rand,
                              h0_off=h0_off,w0_off=w0_off,
                              h1_off=h1_off,w1_off=w1_off)

    # -- batching info --
    n_h0,n_w0 = search.query_batch_info(vid.shape,False,False) # just showing api
    ntotal = t * n_h0 * n_w0
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- query inds --
    qindex = 0

    # -- run my search --
    vid_te = vid.clone()
    vid_te.requires_grad_(True)
    score_te,inds_te = search(vid_te,qindex,ntotal,vid1=vid)
    print(n_h0,n_w0)
    print(score_te.shape)
    score_grad = th.rand_like(score_te)
    th.autograd.backward(score_te,score_grad)
    grad_te = vid_te.grad
    print(th.all(grad_te.abs() < 1e-5))

    # -- run search --
    grad0,grad1 = dnls.simple.search_bwd.run(score_grad,vid,vid,inds_te,qindex,
                                             stride0,ps,pt,dilation,
                                             use_adj,reflect_bounds)
    grad_gt = grad0# + grad1
    print(grad_te[0,0,:3,:3])
    print(grad0[0,0,:3,:3])
    print(grad1[0,0,:3,:3])
    print(grad_gt[0,0,:3,:3])

    # -- viz --
    diff2 = (grad_gt - grad_te)**2
    # diff2 = th.abs(grad_te - grad_gt)/(grad_gt.abs()+1e-5)
    print(diff2.max())
    diff2 /= diff2.max().item()
    rand_s = "rand" if use_rand else "norand"
    fn = "grad_exact_diff_%s" % rand_s
    dnls.testing.data.save_burst(diff2,SAVE_DIR,fn)
    # print(score_te[0,:3])
    # print(score_gt[0,:3])

    # -- compare --
    tol = 1e-5
    error = th.mean(th.abs(grad_te - grad_gt)/(grad_gt.abs()+1e-5)).item()
    if error > tol: print("error: ",error)
    assert error < tol

    tol = 1e-3
    max_error = th.abs((grad_te - grad_gt)/(grad_gt.abs()+1e-5)).max().item()
    if max_error > tol: print("max error: ",max_error)
    assert max_error < tol


def test_cu_full_ws(ps,stride0,stride1,dilation,reflect_bounds,exact):




    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    k,pt = 1,1
    wt = 0
    ws = 15
    # stride0 = stride
    # stride1 = 1
    search_abs = k<=0
    use_k = False#k>0
    if ws == -1 and k > 0: ws = 10
    exact = True
    use_adj = True
    full_ws = True

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    adj = True
    only_full = True

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device)[:1,].contiguous()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-2)
    vid = th.cat([vid,vid],-2)

    # -- normalize --
    vid /= vid.max()
    # vidr = th.ones_like(vid)
    vidr = th.rand_like(vid)

    # -- compute flow --
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape
    chnls = vid.shape[1]

    # -- exec fold fxns --
    h0_off,w0_off,h1_off,w1_off = 0,0,0,0
    search = dnls.search.init("l2_with_index",
                              flows.fflow, flows.bflow, k, ps, pt,
                              ws, wt, dilation=dil,
                              stride0=stride0, stride1=stride1,

                              use_k = use_k,use_adj=use_adj,
                              reflect_bounds=reflect_bounds,
                              search_abs=search_abs,
                              full_ws=full_ws,exact=exact,
                              h0_off=h0_off,w0_off=w0_off,
                              h1_off=h1_off,w1_off=w1_off)

    # -- batching info --
    n_h0,n_w0 = search.query_batch_info(vid.shape) # just showing api
    ntotal = t * n_h0 * n_w0
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- viz for testing --
    # vid_search = dnls.simple.search_full_ws.run(vid,ws,stride0,0,0,0)
    # print(th.sum(vid_search))
    # dnls.testing.data.save_burst(vid_search,"./output","vid_ss_00")
    # vid_search = dnls.simple.search_full_ws.run(vid,ws,stride0,0,128-4,128-4)
    # print(th.sum(vid_search))
    # dnls.testing.data.save_burst(vid_search,"./output","vid_ss_br")
    # vid_search = dnls.simple.search_full_ws.run(vid,ws,stride0,0,127,127)
    # print(th.sum(vid_search))
    # dnls.testing.data.save_burst(vid_search,"./output","vid_ss_-1-1")
    # vid_search = dnls.simple.search_full_ws.run(vid,ws,stride0,0,32,32)
    # print(th.sum(vid_search))
    # dnls.testing.data.save_burst(vid_search,"./output","vid_ss_mid")

    # -- viz ours --
    # print(score_te)
    # print(score_te[0])
    # print(score_te[1])

    # -- testing code --
    qindex = 0
    score_te,inds_te = search(vid,qindex,ntotal,vid1=vidr)
    assert th.all(score_te<th.inf).item()

#
# -- Backward Testing --
#


def test_cu_vs_th_bwd(ps,stride0,stride1,dilation,reflect_bounds,exact):
    """

    Test the CUDA code with torch code

    Backward Pass

    """
    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    k,pt = 1,1
    wt = 0
    ws = -1
    k = -1
    # stride0 = stride
    # stride1 = 4
    search_abs = True
    use_k = k>0
    exact = True

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    adj = True
    only_full = True

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device)[:1,].contiguous()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],-2)

    # -- normalize --
    vid /= vid.max()
    # vidr = th.ones_like(vid)
    vidr = th.rand_like(vid)

    # -- allow for grads --
    vid0_te = vid.clone()
    vid1_te = vidr.clone()
    vid0_gt = vid.clone()
    vid1_gt = vidr.clone()
    vid0_te.requires_grad_(True)
    vid1_te.requires_grad_(True)
    vid0_gt.requires_grad_(True)
    vid1_gt.requires_grad_(True)

    # -- compute flow --
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape
    chnls = vid.shape[1]

    # -- sub square --
    top,btm,left,right = 0,h,0,w
    coords = [top,left,btm,right]
    # sq_h = coords[2] - coords[0]
    # sq_w = coords[3] - coords[1]

    # -- batching --
    _,_,n0,n1 = get_batching_info(vid.shape,stride0,stride1,ps,dil)
    n_h0,n_w0 = n0[0],n0[1]
    n_h1,n_w1 = n1[0],n1[1]

    # -- batching info --
    ntotal = t * n_h0 * n_w0
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- exec fold fxns --
    use_adj = True
    h0_off,w0_off,_,_ = comp_pads(vid.shape, ps, stride0, 1)
    h1_off,w1_off,_,_ = comp_pads(vid.shape, ps, stride1, 1)
    search = dnls.search.init("l2_with_index",
                              flows.fflow, flows.bflow, k, ps, pt,
                              ws, wt, dilation=dil,
                              stride0=stride0, stride1=stride1,
                              use_k = use_k,use_adj=use_adj,
                              reflect_bounds=reflect_bounds,
                              search_abs=search_abs,exact=exact,
                              h0_off=h0_off,w0_off=w0_off,
                              h1_off=h1_off,w1_off=w1_off)

    # -- run search --
    qindex = 0
    score_te,inds_te = search(vid0_te,qindex,ntotal,vid1=vid1_te)
    score_te = rearrange(score_te,'(sh sw) (h w) -> h w sh sw',sh=n_h0,h=n_h1)

    # -- comparison --
    mode = "reflect" if reflect_bounds else "zero"
    score_gt = dnls.simple.search_nn.run_nn(vid0_gt,ps,stride=stride0,mode=mode,
                                            dilation=dil,vid1=vid1_gt,stride1=stride1)
    score_gt = rearrange(score_gt,'(sh sw) (h w) -> h w sh sw',sh=n_h0,h=n_h1)

    # -- compute gradient --
    score_grad = th.rand_like(score_gt)
    th.autograd.backward(score_gt,score_grad)
    th.autograd.backward(score_te,score_grad)

    # -- unpack grads --
    grad0_te = vid0_te.grad
    grad1_te = vid1_te.grad
    grad0_gt = vid0_gt.grad
    grad1_gt = vid1_gt.grad


    # -- viz --
    diff = th.abs((grad0_te - grad0_gt)/(grad0_gt.abs()+1e-5))
    diff /= diff.max()
    dnls.testing.data.save_burst(diff,"./output/tests/test_search_with_index/","grad0")
    diff = th.abs((grad1_te - grad1_gt)/(grad1_gt.abs()+1e-5))
    diff /= diff.max()
    dnls.testing.data.save_burst(diff,"./output/tests/test_search_with_index/","grad1")

    #
    # -- Backward Step --
    #

    # -- tolerances --
    small_thresh = 1e-2
    tol_mean = 1e-4
    tol_max = 5e-3

    # -- check 0 --
    args = th.where(grad0_gt.abs() > small_thresh)
    diff = th.abs((grad0_te - grad0_gt)/(grad0_gt.abs()+1e-5))
    error = diff.mean().item()
    assert error < tol_mean
    error = diff[args].max().item()
    assert error < tol_max

    # -- check 1 --
    args = th.where(grad1_gt.abs() > small_thresh)
    diff = th.abs((grad1_te - grad1_gt)/(grad1_gt.abs()+1e-10))
    error = diff.mean().item()
    assert error < tol_mean
    error = diff[args].max().item()
    assert error < tol_max

