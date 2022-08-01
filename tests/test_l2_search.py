
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

# -- check if reordered --
from scipy import optimize
SAVE_DIR = Path("./output/tests/l2_search")

#
# -- meshgrid --
#

def pytest_generate_tests(metafunc):
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)
    # test_lists = {"ps":[3],"stride":[1],"dilation":[1,2],
    #               "top":[3],"btm":[62],"left":[2],"right":[62]}
    # test_lists = {"ps":[4],"stride":[1,2],"dilation":[2],
    #               "top":[4],"btm":[64],"left":[1],"right":[61]}
    # test_lists = {"ps":[3,4,5,6,7,8],"stride":[1,2,3,4,5],"dilation":[1,2,3,4,5],
    #               "top":[3],"btm":[57],"left":[7],"right":[57]}
    # test_lists = {"ps":[3],"stride":[2],"dilation":[2],
    #               "top":[3],"btm":[57],"left":[7],"right":[57]}
    test_lists = {"ps":[7],"stride0":[4],"stride1":[4],"dilation":[1],"wt":[0],
                  "ws":[-1],"top":[0],"btm":[64],"left":[0],"right":[64],"k":[-1],
                  "exact":[True],"reflect_bounds":[False]}
    # test_lists = {"ps":[3,4,5,6,7,8],"stride":[1,2,3,4,5,8],"dilation":[1,2,3,4,5,8],
    #               "top":[1,11],"btm":[50,57],"left":[3,7],"right":[57,30]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)


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
    k,pt = 1,1
    wt = 0
    ws = -1
    k = -1
    # stride0 = stride
    # stride1 = 1
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

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid.shape, ps, stride0, dil)
    oh1,ow1,hp1,wp1 = comp_pads(vid.shape, ps, stride1, dil)
    n_h = (hp - (ps-1)*dil - 1)//stride0 + 1
    n_w = (wp - (ps-1)*dil - 1)//stride0 + 1
    n_h1 = (hp1 - (ps-1)*dil - 1)//stride1 + 1
    n_w1 = (wp1 - (ps-1)*dil - 1)//stride1 + 1

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h * n_w
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- exec fold fxns --
    use_adj = True
    h0_off,w0_off,_,_ = comp_pads(vid.shape, ps, stride0, 1)
    h1_off,w1_off,_,_ = comp_pads(vid.shape, ps, stride1, 1)
    search = dnls.search.init("l2",flows.fflow, flows.bflow, k, ps, pt,
                              ws, wt, dilation=dil, stride=stride1,
                              use_k = use_k,use_adj=use_adj,
                              reflect_bounds=reflect_bounds,
                              search_abs=search_abs,exact=exact,
                              h0_off=h0_off,w0_off=w0_off,
                              h1_off=h1_off,w1_off=w1_off)

    # -- query inds --
    qindex = 0
    iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch,stride0,
                                                coords,t,device)

    # -- run search --
    mode = "reflect" if reflect_bounds else "zero"
    score_gt = dnls.simple.search_nn.run_nn(vid,ps,stride=stride0,mode=mode,
                                              dilation=dil,vid1=vidr,stride1=stride1)
    score_gt = rearrange(score_gt,'(sh sw) (h w) -> h w sh sw',sh=n_h,h=n_h1)


    # -- testing code --
    score_te,inds_te = search(vid,iqueries,vid1=vidr)
    score_te = rearrange(score_te,'(sh sw) (h w) -> h w sh sw',sh=n_h,h=n_h1)

    # -- compare --
    tol = 1e-5
    error = th.mean(th.abs(score_te - score_gt)/score_gt.abs()).item()
    if error > tol: print("error: ",error)
    assert error < tol

    tol = 1e-4
    max_error = th.abs((score_te - score_gt)/score_gt.abs()).max().item()
    if max_error > tol: print("max error: ",max_error)
    assert max_error < tol

def test_cu_vs_simp_fwd(ps,k,stride0,stride1,dilation,reflect_bounds,exact):


    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    k,pt = 1,1
    wt = 0
    ws = -1
    # stride0 = stride
    # stride1 = 1
    search_abs = k<=0
    use_k = k>0
    if ws == -1 and k > 0: ws = 10
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

    # -- sub square --
    top,btm,left,right = 0,h,0,w
    coords = [top,left,btm,right]
    # sq_h = coords[2] - coords[0]
    # sq_w = coords[3] - coords[1]

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid.shape, ps, stride0, dil)
    oh1,ow1,hp1,wp1 = comp_pads(vid.shape, ps, stride1, dil)
    n_h = (hp - (ps-1)*dil - 1)//stride0 + 1
    n_w = (wp - (ps-1)*dil - 1)//stride0 + 1
    n_h1 = (hp1 - (ps-1)*dil - 1)//stride1 + 1
    n_w1 = (wp1 - (ps-1)*dil - 1)//stride1 + 1

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h * n_w
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- exec fold fxns --
    h0_off,w0_off,_,_ = comp_pads(vid.shape, ps, stride0, 1)
    h1_off,w1_off,_,_ = comp_pads(vid.shape, ps, stride1, 1)
    search = dnls.search.init("l2",flows.fflow, flows.bflow, k, ps, pt,
                              ws, wt, dilation=dil, stride=stride1,
                              use_k = use_k,use_adj=use_adj,
                              reflect_bounds=reflect_bounds,
                              search_abs=search_abs,exact=exact,
                              h0_off=h0_off,w0_off=w0_off,
                              h1_off=h1_off,w1_off=w1_off)

    # -- query inds --
    qindex = 0
    iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch,stride0,
                                                coords,t,device)

    # -- run search --
    score_gt,_ = dnls.simple.search.run(vid,iqueries,flows,k,ps,pt,ws,wt,chnls,
                                        dilation=dil,stride=stride1,use_adj=use_adj,
                                        reflect_bounds=reflect_bounds,
                                        search_abs=search_abs,
                                        h0_off=h0_off,w0_off=w0_off,
                                        h1_off=h1_off,w1_off=w1_off,
                                        vid1=vidr)

    # -- testing code --
    score_te,inds_te = search(vid,iqueries,vid1=vidr)

    # -- viz --
    # print(score_te[0,:3])
    # print(score_gt[0,:3])

    # -- compare --
    tol = 1e-5
    error = th.mean(th.abs(score_te - score_gt)/score_gt.abs()).item()
    if error > tol: print("error: ",error)
    assert error < tol

    tol = 1e-4
    max_error = th.abs((score_te - score_gt)/score_gt.abs()).max().item()
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

    # -- sub square --
    top,btm,left,right = 0,h,0,w
    coords = [top,left,btm,right]
    # sq_h = coords[2] - coords[0]
    # sq_w = coords[3] - coords[1]

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid.shape, ps, stride0, dil)
    oh1,ow1,hp1,wp1 = comp_pads(vid.shape, ps, stride1, dil)
    n_h = (hp - (ps-1)*dil - 1)//stride0 + 1
    n_w = (wp - (ps-1)*dil - 1)//stride0 + 1
    n_h1 = (hp1 - (ps-1)*dil - 1)//stride1 + 1
    n_w1 = (wp1 - (ps-1)*dil - 1)//stride1 + 1

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h * n_w
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- exec fold fxns --
    h0_off,w0_off,_,_ = comp_pads(vid.shape, ps, stride0, 1)
    h1_off,w1_off,_,_ = comp_pads(vid.shape, ps, stride1, 1)
    h0_off,w0_off,h1_off,w1_off = 0,0,0,0
    search = dnls.search.init("l2",flows.fflow, flows.bflow, k, ps, pt,
                              ws, wt, dilation=dil, stride=stride1,
                              use_k = use_k,use_adj=use_adj,
                              reflect_bounds=reflect_bounds,
                              search_abs=search_abs,
                              full_ws=full_ws,exact=exact,
                              h0_off=h0_off,w0_off=w0_off,
                              h1_off=h1_off,w1_off=w1_off)

    # -- query inds --
    qindex = 0
    iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch,stride0,
                                                coords,t,device)
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
    # print(iqueries[:3])
    score_te,inds_te = search(vid,iqueries,vid1=vidr)
    # print(score_te[0])
    # print(th.where(score_te > 10000.))
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

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid.shape, ps, stride0, dil)
    oh1,ow1,hp1,wp1 = comp_pads(vid.shape, ps, stride1, dil)
    n_h = (hp - (ps-1)*dil - 1)//stride0 + 1
    n_w = (wp - (ps-1)*dil - 1)//stride0 + 1
    n_h1 = (hp1 - (ps-1)*dil - 1)//stride1 + 1
    n_w1 = (wp1 - (ps-1)*dil - 1)//stride1 + 1

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h * n_w
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- exec fold fxns --
    use_adj = True
    h0_off,w0_off,_,_ = comp_pads(vid.shape, ps, stride0, 1)
    h1_off,w1_off,_,_ = comp_pads(vid.shape, ps, stride1, 1)
    search = dnls.search.init("l2",flows.fflow, flows.bflow, k, ps, pt,
                              ws, wt, dilation=dil, stride=stride1,
                              use_k = use_k,use_adj=use_adj,
                              reflect_bounds=reflect_bounds,
                              search_abs=search_abs,exact=exact,
                              h0_off=h0_off,w0_off=w0_off,
                              h1_off=h1_off,w1_off=w1_off)

    # -- query inds --
    qindex = 0
    iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch,stride0,
                                                coords,t,device)
    # -- run search --
    score_te,inds_te = search(vid0_te,iqueries,vid1=vid1_te)
    score_te = rearrange(score_te,'(sh sw) (h w) -> h w sh sw',sh=n_h,h=n_h1)

    # -- comparison --
    mode = "reflect" if reflect_bounds else "zero"
    score_gt = dnls.simple.search_nn.run_nn(vid0_gt,ps,stride=stride0,mode=mode,
                                            dilation=dil,vid1=vid1_gt,stride1=stride1)
    score_gt = rearrange(score_gt,'(sh sw) (h w) -> h w sh sw',sh=n_h,h=n_h1)

    # -- compute gradient --
    score_grad = th.rand_like(score_gt)
    th.autograd.backward(score_gt,score_grad)
    th.autograd.backward(score_te,score_grad)

    # -- unpack grads --
    grad0_te = vid0_te.grad
    grad1_te = vid1_te.grad
    grad0_gt = vid0_gt.grad
    grad1_gt = vid1_gt.grad


    #
    # -- Backward Step --
    #

    # -- tolerances --
    small_thresh = 1e-2
    tol_mean = 1e-4
    tol_max = 5*1e-3

    # -- check 0 --
    args = th.where(grad0_gt.abs() > small_thresh)
    diff = th.abs((grad0_te - grad0_gt)/(grad0_gt.abs()+1e-5))
    error = diff.mean().item()
    assert error < tol_mean
    error = diff[args].max().item()
    assert error < tol_max

    # -- check 1 --
    args = th.where(grad1_gt.abs() > small_thresh)
    diff = th.abs((grad1_te - grad1_gt)/(grad1_gt.abs()+1e-5))
    error = diff.mean().item()
    assert error < tol_mean
    error = diff[args].max().item()
    assert error < tol_max


