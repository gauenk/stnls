

# -- data mgnmt --
from pathlib import Path

# -- testing --
import pytest
import random

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- patchify --
from torch.nn.functional import fold,unfold,pad

# -- dnls --
import dnls
import dnls.utils.gpu_mem as gpu_mem
from dnls.utils.pads import comp_pads
from dnls.utils.inds import get_batching_info

# -- paths --
SAVE_DIR = Path("./output/tests/prod_search")

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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
    test_lists = {"ps":[7],"stride":[4],"dilation":[1],"wt":[1],
                  "ws":[3],"top":[0],"btm":[64],"left":[0],"right":[64],"k":[12],
                  "exact":[True],"seed":[123]}
    # test_lists = {"ps":[3,4,5,6,7,8],"stride":[1,2,3,4,5,8],"dilation":[1,2,3,4,5,8],
    #               "top":[1,11],"btm":[50,57],"left":[3,7],"right":[57,30]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def test_fwd(ws,wt,k,ps,stride,dilation,exact):
    """

    Test the CUDA code with torch code

    Forward Pass

    """


    # -- get args --
    dil = dilation
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    pt = 1
    stride0 = stride
    stride1 = 1

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = True
    gpu_stats = False
    adj = False
    reflect_bounds = False

    # -- load data --
    vid = dnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid.to(device)[:,:3,].contiguous()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    vid = th.cat([vid,vid],-1)
    vid = th.cat([vid,vid],-2)
    # print(vid.shape)

    # -- compute flow --
    flows = dnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)
    pflows = dnls.nn.ofa.run(flows,stride0=stride0)
    # pflows.fflow = pflows.bflow
    # pflows.fflow = th.randn_like(pflows.bflow)
    # print(flows.fflow.max(),flows.fflow.min())
    # print(flows.fflow.shape)
    # print(pflows.fflow.shape)
    # print(flows.fflow[0,:,:,0,36])
    # print(pflows.fflow[0,:,0,:,0,36])
    # print(pflows.fflow[0,0,:,:,0,36])

    # -- normalize --
    vid /= vid.max()

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape
    chnls = vid.shape[1]

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid[0].shape, ps, stride0, dil)
    oh1,ow1,_,_ = comp_pads(vid[0].shape, ps, stride1, dil)
    _,_,n0,n1 = get_batching_info(vid[0].shape,stride0,stride1,ps,dil)
    n_h0,n_w0 = n0[0],n0[1]
    n_h1,n_w1 = n1[0],n1[1]

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h0 * n_w0
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- exec fold fxns --
    use_adj = True
    search = dnls.search.init("prod_pf_with_index",pflows.fflow, pflows.bflow,
                              k, ps, pt, ws, wt, oh0, ow0, oh1, ow1,
                              chnls=-1,dilation=dil,
                              stride0=stride0, stride1=stride1,
                              reflect_bounds=reflect_bounds,
                              use_k=False,search_abs=False,
                              use_adj=use_adj,exact=exact)
    search_gt = dnls.search.init("prod_with_index",flows.fflow, flows.bflow,
                                 k, ps, pt, ws, wt, oh0, ow0, oh1, ow1,
                                 chnls=-1,dilation=dil,
                                 stride0=stride0, stride1=stride1,
                                 reflect_bounds=reflect_bounds,
                                 use_k=False,use_adj=use_adj,
                                 search_abs=False,exact=exact)

    # -- query inds --

    # -- run search --
    # vidr = None
    # vidr = 10*th.ones_like(vid)
    # vidr = vid
    vidr = th.rand_like(vid)
    # vidr[th.where(th.abs(vidr) > 0.2)] = 1
    # vidr[th.where(th.abs(vidr) < 1)] = 0
    # print(th.unique(vidr))
    # vid = th.ones_like(vid)
    # vid = th.rand_like(vid)
    # vid[th.where(th.abs(vid) > 0.2)] = 1
    # vid[th.where(th.abs(vid) < 1)] = 0

    # ones = th.ones_like(vid)
    # print("vid.shape: ",vid.shape)
    qindex = 0
    score_te,inds_te = search(vid,qindex,nbatch,vid1=vidr)
    score_gt,inds_gt = search_gt(vid,qindex,nbatch,vid1=vidr)
    args = th.where(th.abs(score_te - score_gt)>1e-6)

    # -- compare inds --
    diff = th.sum(th.abs(1.*(inds_te - inds_gt))).item()
    assert diff < 1e-10

    # -- compare --
    is_inf = th.isinf(score_gt)
    is_nan = th.isnan(score_gt)
    is_invalid = th.logical_or(is_nan,is_inf)
    args0 = th.where(th.logical_not(is_invalid)) # remove all inf
    diff = th.abs(score_te - score_gt) / (score_gt.abs() + 1e-5)
    diff = diff[args0]

    tol = 1e-5
    error = th.mean(diff).item()
    if error > tol: print("error: ",error)
    assert error < tol

    tol = 1e-4
    max_error = th.abs(diff).max().item()
    if max_error > tol: print("max error: ",max_error)
    assert max_error < tol


@pytest.mark.slow
def test_bwd(ps,stride,dilation,exact):
    """

    Test the CUDA code with torch code

    Backward Pass for videos

    """


    # -- get args --
    dil,pt = dilation,1
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    wt = 0
    ws = -1
    k = -1
    stride0 = stride
    stride1 = 1

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    adj = False
    reflect_bounds = False

    # -- load data --
    vid = dnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid.to(device)[:,[4],].contiguous()/255.
    vid = vid + 25./255. * th.randn_like(vid)
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    # vid = th.cat([vid,vid],-3)
    # vid = th.cat([vid,vid],-3)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],-2)
    # print("vid.shape: ",vid.shape)

    # -- compute flow --
    flows = dnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape
    chnls = vid.shape[1]

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid[0].shape, ps, stride0, dil)
    oh1,ow1,_,_ = comp_pads(vid[0].shape, ps, stride1, dil)
    n_h0 = (hp - (ps-1)*dil - 1)//stride0 + 1
    n_w0 = (wp - (ps-1)*dil - 1)//stride0 + 1

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h0 * n_w0
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- swap --
    oh0,ow0,_,_ = comp_pads(vid[0].shape, ps, stride0, dil)
    oh1,ow1,_,_ = comp_pads(vid[0].shape, ps, stride1, dil)
    use_adj = False

    # -- exec fold fxns --
    search = dnls.search.init("prod_with_index",
                              flows.fflow, flows.bflow, k, ps, pt,
                              ws, wt, oh0, ow0, oh1, ow1,
                              chnls=-1,dilation=dil,
                              stride0=stride0,stride1=stride1,
                              reflect_bounds=reflect_bounds,use_k=False,
                              exact=exact,search_abs=True,
                              use_adj = True)

    # -- binary image to remove float error --
    # vidr = None
    # vidr = 10*th.ones_like(vid)
    # vid = th.round(th.rand_like(vid),decimals=2)*100
    # vid = th.rand_like(vid)*1.5
    # vid = th.round(th.rand_like(vid),decimals=10)
    # vidr = th.round(th.rand_like(vid),decimals=3)
    # vidr = th.round(th.rand_like(vid),decimals=3)
    # vid = th.round(th.rand_like(vid),decimals=2)*100.
    # vidr = th.round(th.rand_like(vid),decimals=2)*100.
    # vid = vid.type(th.float32)
    # vidr = vidr.type(th.float32)
    # vidr[th.where(th.abs(vidr) > 0.2)] = 1
    # vidr[th.where(th.abs(vidr) < 1)] = 0
    # # vid = th.ones_like(vid)
    # vid = th.rand_like(vid)
    # vid[th.where(th.abs(vid) > 0.2)] = 1
    # vid[th.where(th.abs(vid) < 1)] = 0

    # vid = vidr.clone()
    # vid[:,:,:3,:3] = 0
    # vid[:,:,0,0] = 0
    # vidr[:,:,:3,:3] = 0

    # -- allow grads --
    vid_te = vid.clone()
    vid_gt = vid.clone()
    vidr_te = vid_te.clone()
    vidr_gt = vid_gt.clone()
    vid_te.requires_grad_(True)
    vid_gt.requires_grad_(True)
    vidr_te.requires_grad_(True)
    vidr_gt.requires_grad_(True)

    #
    # -- run search --
    #

    # -- run cu --
    qindex = 0
    score_te,inds_te = search(vid_te,qindex,nbatch,vid1=vidr_te)
    # score_te = rearrange(score_te,'(sh sw) (h w) -> h w sh sw',sh=n_h0,h=h)

    # -- run nn --
    mode = "reflect" if reflect_bounds else "zero"
    B = vid_gt.shape[0]
    score_gt = []
    for b in range(B):
        score_gt_b,_ = dnls.simple.prod_search_nn.run_nn(vid_gt[b],ps,stride=stride0,
                                                         dilation=dil,vid1=vidr_gt[b],
                                                         mode=mode)
        score_gt.append(score_gt_b)
    score_gt = th.stack(score_gt)
    score_gt = rearrange(score_gt,'b h w sh sw -> b (sh sw) (h w)')
    # print("score_gt.shape: ",score_gt.shape)

    # print(score_gt[0][:3,:3])
    # print(score_te[0][:3,:3])

    # -- vis --
    # diff = th.abs(score_te - score_gt)
    # args = th.where(diff>1e-10)
    # for i in range(len(args)):
    #     print(i,th.unique(args[i]))
    # if diff.max() > 1e-10: diff /= diff.max()
    # dnls.testing.data.save_burst(diff[0,0][None,None],"./output/tests/prod_search/","diff")
    # dnls.testing.data.save_burst(diff[:,:,0,0][None,None],"./output/tests/prod_search/","diff_d00")

    # -- compare fwd --
    max_error = th.abs(score_te - score_gt).max().item()
    # print("max error: ",max_error)
    assert max_error < 1e-3

    error = th.mean(th.abs(score_te - score_gt)).item()
    # print("error: ",error)
    assert error < 1e-4

    # -- compute grad --
    score_grad = th.rand_like(score_gt)/1000.
    th.autograd.backward(score_gt,score_grad)
    th.autograd.backward(score_te,score_grad)

    # -- for both grads --
    _grads_te = [vid_te.grad,vidr_te.grad]
    _grads_gt = [vid_gt.grad,vidr_gt.grad]
    for idx,(grads_te,grads_gt) in enumerate(zip(_grads_te,_grads_gt)):

        # -- viz [the error map looks weird] --
        # print(grads_te[0,-1,-10:,-10:])
        # print(grads_gt[0,-1,-10:,-10:])
        # diff = (grads_te -grads_gt).abs()/(grads_gt.abs()+1e-8)
        # diff /= diff.max()
        # dnls.testing.data.save_burst(diff[:,[0]],SAVE_DIR,"grad_diff_0_%d" % exact)
        # dnls.testing.data.save_burst(diff[:,[1]],SAVE_DIR,"grad_diff_1_%d" % exact)
        # dnls.testing.data.save_burst(diff[:,[2]],SAVE_DIR,"grad_diff_2_%d" % exact)
        # print(idx)

        # -- compare grads --
        rel_error = th.abs(grads_gt - grads_te)/(th.abs(grads_gt)+1e-10)
        rel_error_nz  = rel_error

        tol = 1e-3
        error = th.max(rel_error_nz).item()
        if error > tol: print("Max Error: ",error)
        # print("Max Error: ",error)
        assert error < tol

        tol = 1e-4
        error = th.mean(rel_error_nz).item()
        if error > tol: print("Mean Error: ",error)
        # print("Mean Error: ",error)
        assert error < tol
