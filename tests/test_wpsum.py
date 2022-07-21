"""

Weighted-Patch (Inplace) Sum


Verbose Psuedo-Code:
   yi = softmax(dists)
   patches_i = scatter(b2,nlInds_cu).type(th.float64)
   patches_i = rearrange(patches_i,'n k 1 c h w -> n k (c h w)')
   zi = th.sum(yi * patches_i,1).type(th.float32) # n (c h w)

"""


# -- python --
import sys

# -- data mgnmt --
from pathlib import Path
from easydict import EasyDict as edict

# -- testing --
import unittest,pytest

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- dnls --
import dnls
import dnls.utils.gpu_mem as gpu_mem
from dnls.utils.pads import same_padding,comp_pads


# -- meshgrid --
import cache_io

# -- test func --
from torch.nn.functional import fold,unfold,pad,softmax,log_softmax
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/wpsum/")

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)

def pytest_generate_tests(metafunc):
    seed = 123
    set_seed(seed)
    # test_lists = {"ps":[3],"stride":[1],"dilation":[1,2],
    test_lists = {"ps":[7],"stride":[4],"dilation":[1],"wt":[3],"k":[-1],
                  "ws":[10],"top":[0],"btm":[-1],"left":[0],"right":[-1],
                  "exact":[True]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def simple_run(vid,scores_s,inds,ps,pt,reflect_bounds,exact):
    scatter = dnls.scatter.ScatterNl(ps,pt,exact=exact,reflect_bounds=reflect_bounds)
    patches_i = scatter(vid,inds).type(th.float64)
    patches_i = rearrange(patches_i,'n k 1 c h w -> n k (c h w)')
    scores_s = scores_s[...,None].type(th.float64)
    wpatches_i = th.sum(scores_s * patches_i,1).type(th.float32)
    return wpatches_i

def test_forward(ps,stride,dilation,top,btm,left,right,k,exact):

    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    chnls,pt = 1,1
    stride0 = stride
    stride1 = 1
    ws = -1 if k == -1 else 10
    wt = 0 if k == -1 else 5

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    reflect_bounds = False
    use_search_abs = ws == -1
    use_k = k != -1
    use_unfold = k == -1
    t = 1 if use_unfold else 3
    adj = ps//2 if use_unfold else 0

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device)[:t,].contiguous()/255.
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],0)
    # vid = th.cat([vid,vid],0)
    # print("vid.shape: ",vid.shape)

    # -- compute flow --
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape

    # -- sub square --
    top,btm,left,right = 0,h,0,w
    coords = [top,left,btm,right]

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid.shape, ps, stride0, dil)
    oh1,ow1,__,__ = comp_pads(vid.shape, ps, stride1, dil)
    n_h = (hp - (ps-1)*dil - 1)//stride0 + 1
    n_w = (wp - (ps-1)*dil - 1)//stride0 + 1

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h * n_w
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- init xsearch --
    xsearch = dnls.xsearch.CrossSearchNl(flows.fflow, flows.bflow,
                                         k, ps, pt, ws, wt, oh0, ow0, oh1, ow1,
                                         dilation=dil, stride=stride1, use_k=use_k,
                                         reflect_bounds=reflect_bounds, use_search_abs=use_search_abs)

    # -- query inds --
    qindex = 0
    iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch,stride0,
                                                coords,t,device)

    # -- init our inner product --
    h_off,w_off = oh1,ow1
    if not(use_unfold): h_off,w_off = 0,0
    adj,h_off,w_off = 0,0,0
    wpsum = dnls.wpsum.WeightedPatchSum(ps, pt, h_off=h_off,w_off=w_off, dilation=dil,
                                        reflect_bounds=reflect_bounds, adj=adj, exact=exact)

    # -- run xsearch & wpsum --
    scores,inds = xsearch(vid,iqueries,vid1=vid)
    scores_s = softmax(scores*10,dim=1)
    wpatches_te = wpsum(vid,scores_s,inds).view(iqueries.shape[0],-1)

    # -- run simple forward for testing --
    if use_unfold:
        mode = "reflect" if reflect_bounds else "zero"
        scores_gt,wpatches_gt = dnls.simple.xsearch_nn.run_nn(vid,ps,stride=stride0,dilation=dil,vid1=vid,mode=mode)
        wpatches_gt = wpatches_gt.view(iqueries.shape[0],-1)
    else:
        wpatches_gt = simple_run(vid,scores_s,inds,ps,pt,reflect_bounds,exact)

    # -- vis [scores] --
    # scores = scores.view(n_h,n_w,h,w)
    # scores = rearrange(scores,'nh nw h w -> h w nh nw')
    # scores_gt = scores_gt#.view(n_h,n_w,h,w)
    # print(scores[0,0,:3,:3])
    # print(scores_gt[0,0,:3,:3])

    # -- vis --
    # print(wpatches_te[:3,:3])
    # print(wpatches_gt[:3,:3])
    # diff = th.abs(wpatches_gt - wpatches_te).mean(-1)
    # diff = rearrange(diff,'(h w) -> 1 1 h w',h=32)
    # diff /= diff.max()
    # dnls.testing.data.save_burst(diff,SAVE_DIR,"diff_%d" % use_unfold)

    # -- compare --
    tol = 1e-5 if use_unfold else 1e-7
    error = th.abs(wpatches_gt - wpatches_te).mean().item()
    if error > tol: print(error)
    assert error < tol

    tol = 1e-4 if use_unfold else 1e-6
    error = th.abs(wpatches_gt - wpatches_te).max().item()
    if error > tol: print(error)
    assert error < tol


def test_score_backward(ps,stride,dilation,top,btm,left,right,k):

    # -- get args --
    pt,dil = 1,dilation
    dname,ext = "davis_baseball_64x64","jpg"
    stride0 = stride
    stride1 = 1
    ws = -1 if k == -1 else 10
    wt = 0 if k == -1 else 5

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    reflect_bounds = False
    use_search_abs = ws == -1
    use_k = k != -1
    use_unfold = k == -1
    t = 1 if use_unfold else 3
    adj = ps//2 if use_unfold else 0
    exact = True

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)/255.
    h,w = 64,64
    vid = th.from_numpy(vid).to(device)[[4],:,:h,:w].contiguous()
    vid = vid + 25./255 * th.randn_like(vid)
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],0)
    # vid = th.cat([vid,vid],0)
    # print("vid.shape: ",vid.shape)

    # -- compute flow --
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape

    # -- sub square --
    top,btm,left,right = 0,h,0,w
    coords = [top,left,btm,right]

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid.shape, ps, stride0, dil)
    oh1,ow1,_,_ = comp_pads(vid.shape, ps, stride1, dil)
    n_h = (hp - (ps-1)*dil - 1)//stride0 + 1
    n_w = (wp - (ps-1)*dil - 1)//stride0 + 1

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h * n_w
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- init xsearch --
    xsearch = dnls.xsearch.CrossSearchNl(flows.fflow, flows.bflow, k, ps, pt,
                                         ws, wt, oh0, ow0, oh1, ow1,
                                         dilation=dil, stride=stride1, reflect_bounds=reflect_bounds,
                                         use_k=use_k,use_search_abs=use_search_abs,exact=exact)

    # -- query inds --
    qindex = 0
    iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch,stride0,
                                                coords,t,device)

    # -- init our inner product --
    h_off,w_off = oh1,ow1
    if not(use_unfold): h_off,w_off = 0,0
    adj,h_off,w_off = 0,0,0
    wpsum = dnls.wpsum.WeightedPatchSum(ps, pt, h_off=h_off, w_off=w_off, dilation=dil,
                                        reflect_bounds=reflect_bounds, adj=adj, exact=exact)

    # -- run search --

    # -- prepare for grads --
    # vid = th.rand_like(vid)
    vid0_te = vid.clone()
    vid0_gt = vid.clone()
    # vid0_te.requires_grad_(True)
    # vid0_gt.requires_grad_(True)

    vid1_te = vid.clone()#th.randn_like(vid)
    vid1_gt = vid1_te.clone()
    # vid1_te.requires_grad_(True)
    # vid1_gt.requires_grad_(True)

    vid2_te = vid.clone()#th.randn_like(vid)
    vid2_gt = vid2_te.clone()
    # vid2_te.requires_grad_(True)
    # vid2_gt.requires_grad_(True)

    # -- forward passes --
    # scores,inds = xsearch(vid0_te,iqueries,vid1=vid1_te)
    # scores_s = softmax(scores*10,dim=1)
    # # scores_s = scores_s.detach()
    # wpatches_te = wpsum(vid2_te,scores_s,inds).view(iqueries.shape[0],-1)


    #
    # -- forward pass [v2] --
    #

    # -- inds --
    # vid0_te.requires_grad_(False)
    _,inds = xsearch(vid0_te,iqueries,vid1=vid1_te)
    # vid0_te.requires_grad_(True)

    # -- scores --
    mode = "reflect" if reflect_bounds else "zero"
    # vid2_te.requires_grad_(False)
    scores_te,wpatches_te = dnls.simple.xsearch_nn.run_nn(vid0_te,ps,stride=stride0,dilation=dil,vid1=vid1_te,vid2=vid2_te,mode=mode)
    wpatches_te = wpatches_te.view(iqueries.shape[0],-1)
    scores_te = rearrange(scores_te,'h w nh nw -> (nh nw) (h w)')
    scores_te.requires_grad_(True)
    scores_s_te = softmax(scores_te*10,dim=1)
    scores_s_te.requires_grad_(True)
    scores_s_te.retain_grad()

    # -- patches --
    wpatches_te = wpsum(vid2_te,scores_s_te,inds).view(iqueries.shape[0],-1)

    # -- run simple forward for testing --
    mode = "reflect" if reflect_bounds else "zero"
    scores_gt,scores_s_gt,wpatches_gt = dnls.simple.xsearch_nn.run_nn_v2(vid0_gt,ps,stride=stride0,
                                                                         dilation=dil,vid1=vid1_gt,vid2=vid2_gt,mode=mode)
    wpatches_gt = wpatches_gt.view(iqueries.shape[0],-1)

    # -- confirm fwd --
    tol = 1e-5 if use_unfold else 1e-7
    error = th.abs(wpatches_te - wpatches_gt).mean().item()
    if error > tol: print(error)
    assert error < tol

    # -- backward passes --
    wpatches_grad = th.rand_like(wpatches_te)
    th.autograd.backward(wpatches_te,wpatches_grad)
    th.autograd.backward(wpatches_gt,wpatches_grad)

    # -- grab grads --
    _grads_te = [scores_te.grad,scores_s_te.grad]
    _grads_gt = [scores_gt.grad,scores_s_gt.grad]
    for idx,(grads_te,grads_gt) in enumerate(zip(_grads_te,_grads_gt)):

        print(grads_te[:3,:3])
        print(grads_gt[:3,:3])
        args = th.where(grads_gt.abs() > 1e-5)
        print(grads_te[args][:5])
        print(grads_gt[args][:5])

        error = th.abs((grads_te - grads_gt)/(grads_gt.abs()+1e-10)).mean().item()
        print("mean error: ",error)

        error = th.abs((grads_te - grads_gt)/(grads_gt.abs()+1e-10)).max().item()
        print("max error: ",error)


def test_vid_backward(ps,stride,dilation,top,btm,left,right,k):

    # -- get args --
    pt,dil = 1,dilation
    dname,ext = "davis_baseball_64x64","jpg"
    stride0 = stride
    stride1 = 1
    ws = -1 if k == -1 else 10
    wt = 0 if k == -1 else 5

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    reflect_bounds = False
    use_search_abs = ws == -1
    use_k = k != -1
    use_unfold = k == -1
    t = 1 if use_unfold else 3
    adj = ps//2 if use_unfold else 0
    exact = True

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)/255.
    vid = th.from_numpy(vid).to(device)[[4],].contiguous()
    vid = vid + 25./255 * th.randn_like(vid)
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],0)
    # vid = th.cat([vid,vid],0)
    # print("vid.shape: ",vid.shape)

    # -- compute flow --
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape

    # -- sub square --
    top,btm,left,right = 0,h,0,w
    coords = [top,left,btm,right]

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid.shape, ps, stride0, dil)
    oh1,ow1,_,_ = comp_pads(vid.shape, ps, stride1, dil)
    n_h = (hp - (ps-1)*dil - 1)//stride0 + 1
    n_w = (wp - (ps-1)*dil - 1)//stride0 + 1

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h * n_w
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- init xsearch --
    xsearch = dnls.xsearch.CrossSearchNl(flows.fflow, flows.bflow, k, ps, pt,
                                         ws, wt, oh0, ow0, oh1, ow1,
                                         dilation=dil, stride=stride1, reflect_bounds=reflect_bounds,
                                         use_k=use_k,use_search_abs=use_search_abs,exact=exact)

    # -- query inds --
    qindex = 0
    iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch,stride0,
                                                coords,t,device)

    # -- init our inner product --
    h_off,w_off = oh1,ow1
    if not(use_unfold): h_off,w_off = 0,0
    adj,h_off,w_off = 0,0,0
    wpsum = dnls.wpsum.WeightedPatchSum(ps, pt, h_off=h_off, w_off=w_off, dilation=dil,
                                        reflect_bounds=reflect_bounds, adj=adj, exact=exact)

    # -- run search --

    # -- prepare for grads --
    # vid = th.rand_like(vid)
    vid0_te = vid.clone()
    vid0_gt = vid.clone()
    vid0_te.requires_grad_(True)
    vid0_gt.requires_grad_(True)

    vid1_te = vid.clone()#th.randn_like(vid)
    vid1_gt = vid1_te.clone()
    vid1_te.requires_grad_(True)
    vid1_gt.requires_grad_(True)

    vid2_te = vid.clone()#th.randn_like(vid)
    vid2_gt = vid2_te.clone()
    vid2_te.requires_grad_(True)
    vid2_gt.requires_grad_(True)

    # -- forward passes --
    # scores,inds = xsearch(vid0_te,iqueries,vid1=vid1_te)
    # scores_s = softmax(scores*10,dim=1)
    # # scores_s = scores_s.detach()
    # wpatches_te = wpsum(vid2_te,scores_s,inds).view(iqueries.shape[0],-1)


    #
    # -- forward pass [v2] --
    #

    # -- inds --
    vid0_te.requires_grad_(False)
    _,inds = xsearch(vid0_te,iqueries,vid1=vid1_te)
    vid0_te.requires_grad_(True)


    # -- scores --
    mode = "reflect" if reflect_bounds else "zero"
    vid2_te.requires_grad_(False)
    scores_te,wpatches_te = dnls.simple.xsearch_nn.run_nn(vid0_te,ps,stride=stride0,dilation=dil,vid1=vid1_te,vid2=vid2_te,mode=mode)
    wpatches_te = wpatches_te.view(iqueries.shape[0],-1)

    scores_te = rearrange(scores_te,'h w nh nw -> (nh nw) (h w)')
    scores_s = softmax(scores_te*10,dim=1)
    scores_s = scores_s.view(iqueries.shape[0],-1)

    # -- patches --
    vid2_te.requires_grad_(True)
    wpatches_te = wpsum(vid2_te,scores_s,inds).view(iqueries.shape[0],-1)



    # -- checking --
    # error = th.abs(scores_s_0 - scores_s).mean()
    # print(error)
    # error = th.abs(scores_s_0 - scores_s).max()
    # print(error)
    # error = th.abs(scores_0 - scores_te).mean()
    # print(error)
    # error = th.abs(scores_0 - scores_te).max()
    # print(error)
    # exit(0)


    # -- run simple forward for testing --
    soft_score = None
    if use_unfold:
        mode = "reflect" if reflect_bounds else "zero"
        scores_gt,wpatches_gt = dnls.simple.xsearch_nn.run_nn(vid0_gt,ps,stride=stride0,dilation=dil,vid1=vid1_gt,vid2=vid2_gt,mode=mode)
        wpatches_gt = wpatches_gt.view(iqueries.shape[0],-1)
    else:
        scores_gt,inds = xsearch(vid_gt_0,iqueries,vid1=vid_gt_0)
        soft_score = th.nn.functional.softmax(scores_gt*10,1)
        wpatches_gt = simple_run(vid_gt,soft_score,inds,ps,pt,reflect_bounds,exact)

    # -- viz --
    print(wpatches_te[:3,:3])
    print(wpatches_gt[:3,:3])


    # -- confirm fwd --
    tol = 1e-5 if use_unfold else 1e-7
    error = th.abs(wpatches_te - wpatches_gt).mean().item()
    if error > tol: print(error)
    assert error < tol

    # -- backward passes --
    wpatches_grad = th.rand_like(wpatches_te)
    th.autograd.backward(wpatches_te,wpatches_grad)
    th.autograd.backward(wpatches_gt,wpatches_grad)

    # -- grab grads --
    vids_te = [vid0_te,vid1_te,vid2_te]
    vids_gt = [vid0_gt,vid1_gt,vid2_gt]
    for idx,(vid_te,vid_gt) in enumerate(zip(vids_te,vids_gt)):

        # -- unpack grads --
        grad_te = vid_te.grad
        grad_gt = vid_gt.grad
        if grad_gt is None: continue
        # print("testing: %d" % idx)

        # -- viz --
        args_te = th.where(grad_te.abs()>1e-2)
        args_gt = th.where(grad_gt.abs()>1e-2)
        # print(grad_te[args_te][:3])
        # print(grad_gt[args_gt][:3])

        # -- get tolerance --
        if exact: tol = 1e-3 if use_unfold else 1e-7
        else: tol = 1e-2 if use_unfold else 1e-7

        # -- viz --
        diff = (grad_te - grad_gt).abs()/(grad_gt.abs()+1e-10)
        args = th.where(grad_gt.abs() < 1e-2)
        args = th.where(diff > 1e-3)
        print(grad_te[args][:5])
        print(grad_gt[args][:5])
        print(diff.mean().item())
        print(diff.max().item())
        Z = 1e-5 if idx > 0 else 1e-2
        # diff /= diff.max()
        diff /= Z
        if idx ==0: diff = diff.clamp(0.,1.)
        dnls.testing.data.save_burst(diff,SAVE_DIR,"grad_diff_%d_%d" % (use_unfold,idx))

        # -- compare --
        error = th.abs((grad_te - grad_gt)).mean().item()
        # print(error)
        if error > tol: print(error)
        assert error < tol

        tol = tol*10 # max is x10 bigger
        error = th.abs((grad_te - grad_gt)).max().item()
        # print(error)
        if error > tol: print(error)
        assert error < tol

