"""

Weighted-Patch (Inplace) Sum "Heads"


Verbose Psuedo-Code:

   patches_i = unfold_k(b2,nlInds_cu).type(th.float64)
   patches_i = rearrange(patches_i,'n k 1 c h w -> n k (c h w)')
   zpatches = []
   for ki in range(k):
      yi = softmax(dists[ki])
      zi = th.sum(yi * patches_i,1).type(th.float32) # n (c h w), this code!
      zpatches.append(zi)
   zpatches = th.stack(zpatches)

"""


# -- python --
import sys

# -- data mgnmt --
from pathlib import Path
from easydict import EasyDict as edict

# -- testing --
import pytest

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- dnls --
import dnls
import dnls.utils.gpu_mem as gpu_mem
from dnls.utils.pads import same_padding,comp_pads

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

def scores_to_heads_batch(scores,nheads,seed=None):
    B = scores.shape[0]
    sheads = []
    for b in range(B):
        sheads_b = scores_to_heads(scores[b],nheads,seed)
        sheads.append(sheads_b)
    sheads = th.stack(sheads)
    return sheads

def scores_to_heads(scores,nheads,seed=None):
    if not(seed is None):
        th.manual_seed(seed)
    score_heads = []
    for head in range(nheads):
        scores_h = softmax(scores*10  + th.randn_like(scores) ,1)
        score_heads.append(scores_h)
    score_heads = th.stack(score_heads,0) # nheads a b
    return score_heads

def simple_run_batch(vid,score_heads,inds,ps,pt,reflect_bounds,exact):
    B = vid.shape[0]
    wpatches = []
    # print("score_heads.shape,inds.shape: " ,score_heads.shape,inds.shape)
    for b in range(B):
        wpatches_b = simple_run(vid[[b]],score_heads[b],
                                inds[[b]],ps,pt,reflect_bounds,exact)
        wpatches.append(wpatches_b)
    wpatches = th.stack(wpatches)
    return wpatches

def simple_run(vid,score_heads,inds,ps,pt,reflect_bounds,exact):
    unfold_k = dnls.UnfoldK(ps,pt,exact=exact,reflect_bounds=reflect_bounds)
    patches_i = unfold_k(vid,inds).type(th.float64)[0]
    H = score_heads.shape[0]
    patches_i = rearrange(patches_i,'q k 1 (H c) h w -> q k H (c h w)',H=H)
    # print("score_heads.shape:" ,score_heads.shape)
    score_heads = rearrange(score_heads,'H q k -> q k H 1')
    # print("score_heads.shape:" ,score_heads.shape)
    # print("patches_i.shape:" ,patches_i.shape)
    wpatches_i = th.sum(score_heads * patches_i,1).type(th.float32)
    wpatches_i = wpatches_i.transpose(1,0).contiguous()
    return wpatches_i

def test_forward(ps,stride,dilation,top,btm,left,right,k,exact):

    # -- get args --
    dil = dilation
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    chnls,pt = 1,1
    stride0 = stride
    stride1 = 1
    ws = -1 if k == -1 else 10
    wt = 0 if k == -1 else 5
    nheads = 5

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    reflect_bounds = False
    search_abs = ws == -1
    use_k = k != -1
    use_unfold = False
    t = 1 if use_unfold else 1
    adj = ps//2 if use_unfold else 0

    # -- load data --
    vid = dnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)/255.
    vid = vid.to(device)[:,:t,].contiguous()
    vid = vid[...,:32,:32].contiguous()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    vid = th.cat([vid,]*nheads,2)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],0)
    # vid = th.cat([vid,vid],0)
    # print("vid.shape: ",vid.shape)

    # -- compute flow --
    flows = dnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape

    # -- sub square --
    top,btm,left,right = 0,h,0,w
    coords = [top,left,btm,right]

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid[0].shape, ps, stride0, dil)
    oh1,ow1,__,__ = comp_pads(vid[0].shape, ps, stride1, dil)
    n_h = (hp - (ps-1)*dil - 1)//stride0 + 1
    n_w = (wp - (ps-1)*dil - 1)//stride0 + 1

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h * n_w
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- init xsearch --
    search = dnls.search.init("prod_with_index",flows.fflow, flows.bflow,
                              k, ps, pt, ws, wt, oh0, ow0, oh1, ow1,
                              dilation=dil, stride0=stride0, stride1=stride1,
                              use_k=use_k,
                              reflect_bounds=reflect_bounds,
                              search_abs=search_abs)

    # -- init our inner product --
    h_off,w_off = oh1,ow1
    if not(use_unfold): h_off,w_off = 0,0
    adj,h_off,w_off = 0,0,0
    wpsum = dnls.reducers.WeightedPatchSumHeads(ps, pt, h_off=h_off,w_off=w_off,
                                                dilation=dil,
                                                reflect_bounds=reflect_bounds,
                                                adj=adj, exact=exact)

    # -- run search --
    qindex = 0
    scores,inds = search(vid,qindex,nbatch,vid1=vid)
    scores_s = scores_to_heads_batch(scores,nheads)
    # print("score_heads.shape: ",scores_s.shape)

    # -- two methods for comparison --
    nq = nbatch
    # print("vid.shape:" ,vid.shape)
    wpatches_te = wpsum(vid,scores_s,inds).view(b,nheads,nq,-1)
    wpatches_gt = simple_run_batch(vid,scores_s,inds,ps,pt,reflect_bounds,exact)

    # -- vis [scores] --
    # scores = scores.view(n_h,n_w,h,w)
    # scores = rearrange(scores,'nh nw h w -> h w nh nw')
    # scores_gt = scores_gt#.view(n_h,n_w,h,w)
    # print(scores[0,0,:3,:3])
    # print(scores_gt[0,0,:3,:3])

    # -- vis --
    # print(wpatches_te.shape)
    # print(wpatches_gt.shape)
    # print(wpatches_te[0,:3,:3])
    # print(wpatches_gt[0,:3,:3])
    # diff = th.abs(wpatches_gt - wpatches_te).mean(-1)
    # diff = rearrange(diff,'(h w) -> 1 1 h w',h=32)
    # diff /= diff.max()
    # dnls.testing.data.save_burst(diff,SAVE_DIR,"diff_%d" % use_unfold)

    # -- compare --
    tol = 1e-7
    error = th.abs(wpatches_gt - wpatches_te).mean().item()
    if error > tol: print(error)
    assert error < tol

    tol = 1e-6
    error = th.abs(wpatches_gt - wpatches_te).max().item()
    if error > tol: print(error)
    assert error < tol


def test_score_backward(ps,stride,dilation,top,btm,left,right,k):

    # -- get args --
    pt,dil = 1,dilation
    ext = "jpg"
    dnames = ["davis_baseball_64x64"]
    stride0 = stride
    stride1 = 1
    ws = -1 if k == -1 else 10
    wt = 0 if k == -1 else 5
    nheads = 5
    seed = 123

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    reflect_bounds = False
    search_abs = ws == -1
    use_k = k != -1
    use_unfold = k == -1
    t = 1 if use_unfold else 1
    adj = ps//2 if use_unfold else 0
    exact = True

    # -- load data --
    h,w = 32,32
    vid = dnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)/255.
    vid = vid.to(device)[:,[0],:,:h,:w].contiguous()
    vid = vid + 25./255 * th.randn_like(vid)
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    vid = th.cat([vid,]*nheads,-3)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],0)
    # vid = th.cat([vid,vid],0)
    # print("vid.shape: ",vid.shape)

    # -- compute flow --
    flows = dnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape

    # -- sub square --
    top,btm,left,right = 0,h,0,w
    coords = [top,left,btm,right]

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid[0].shape, ps, stride0, dil)
    oh1,ow1,_,_ = comp_pads(vid[0].shape, ps, stride1, dil)
    n_h = (hp - (ps-1)*dil - 1)//stride0 + 1
    n_w = (wp - (ps-1)*dil - 1)//stride0 + 1

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h * n_w
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- init search --
    search = dnls.search.init("prod_with_index",flows.fflow, flows.bflow, k, ps, pt,
                              ws, wt, oh0, ow0, oh1, ow1,
                              dilation=dil, stride0=stride0, stride1=stride1,
                              reflect_bounds=reflect_bounds,
                              use_k=use_k,search_abs=search_abs,exact=exact)

    # -- init our inner product --
    h_off,w_off = oh1,ow1
    if not(use_unfold): h_off,w_off = 0,0
    adj,h_off,w_off = 0,0,0
    wpsum = dnls.reducers.WeightedPatchSumHeads(ps, pt, h_off=h_off, w_off=w_off,
                                                dilation=dil,
                                                reflect_bounds=reflect_bounds,
                                                adj=adj, exact=exact)

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

    #
    # -- forward pass --
    #

    # -- compute score --
    qindex = 0
    scores,inds = search(vid0_te,qindex,nbatch,vid1=vid1_te)
    # print("scores.shape: ",scores.shape)
    scores_te = scores.clone()
    scores_gt = scores.clone()
    scores_te.requires_grad_(True)
    scores_gt.requires_grad_(True)
    scores_s_te = scores_to_heads_batch(scores_te,nheads,seed)
    scores_s_gt = scores_to_heads_batch(scores_gt,nheads,seed)
    scores_s_te.retain_grad()
    scores_s_gt.retain_grad()

    # -- forward test --
    # print("scores_s_te.shape: ",scores_s_te.shape)
    nq = nbatch
    wpatches_te = wpsum(vid2_te,scores_s_te,inds).view(b,nheads,nq,-1)

    # -- forward gt --
    wpatches_gt = simple_run_batch(vid2_gt,scores_s_gt,inds,ps,pt,reflect_bounds,exact)

    # -- viz --
    # print(wpatches_te.shape)
    # print(wpatches_gt.shape)
    # print(wpatches_te.shape)
    # print(wpatches_gt.shape)

    # -- confirm fwd --
    tol = 1e-5 if use_unfold else 1e-7
    error = th.abs(wpatches_te - wpatches_gt).mean().item()
    if error > tol: print(error)
    assert error < tol
    th.cuda.synchronize()

    # -- backward passes --
    wpatches_grad = th.rand_like(wpatches_te)
    th.autograd.backward(wpatches_te,wpatches_grad)
    th.cuda.synchronize()
    th.autograd.backward(wpatches_gt,wpatches_grad)
    th.cuda.synchronize()

    # -- set tol --
    tol_mean = 1e-5
    tol_max = 1e-3

    # -- viz --
    # print("grads:")
    # print(scores_s_te.grad.shape)
    # print(scores_s_gt.grad.shape)
    # print(scores_s_te.grad[0,0,:,0])
    # print(scores_s_gt.grad[0,0,:,0])
    # print("-"*30)

    # -- grab grads --
    _grads_te = [scores_te.grad,scores_s_te.grad]
    _grads_gt = [scores_gt.grad,scores_s_gt.grad]
    for idx,(grads_te,grads_gt) in enumerate(zip(_grads_te,_grads_gt)):

        # -- compute error --
        # print("grads_te.shape,grads_gt.shape: ",grads_te.shape,grads_gt.shape)
        diff = th.abs((grads_te - grads_gt)/(grads_gt.abs()+1e-10))
        args = th.where(grads_gt.abs() > 1e-1)

        # -- viz --
        # print(len(args[0]),len(grads_gt.ravel()),grads_gt.abs().mean())
        args2 = th.where(diff[args] > 1e-4)

        # -- compare --
        error = diff.mean().item()
        if exact:
            if error > tol_mean: print("mean error: ",error)
            assert error < tol_mean

        error = diff[args].max().item()
        if exact:
            if error > tol_max: print("max error: ",error)
            assert error < tol_max


@pytest.mark.slow
def test_vid_backward(ps,stride,dilation,top,btm,left,right,k):

    # -- get args --
    pt,dil = 1,dilation
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    stride0 = stride
    stride1 = 1
    ws = -1 if k == -1 else 10
    wt = 0 if k == -1 else 5
    nheads = 5

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    reflect_bounds = False
    search_abs = ws == -1
    use_k = k != -1
    use_unfold = k == -1
    t = 1 if use_unfold else 3
    adj = ps//2 if use_unfold else 0
    exact = True

    # -- load data --
    vid = dnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)/255.
    vid = vid.to(device)[:,[0],:,:32,:32].contiguous()
    vid = vid + 25./255 * th.randn_like(vid)
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    vid = th.cat([vid,]*nheads,-3)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],-2)
    # vid = th.cat([vid,vid],0)
    # vid = th.cat([vid,vid],0)
    # print("vid.shape: ",vid.shape)

    # -- compute flow --
    flows = dnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape

    # -- sub square --
    top,btm,left,right = 0,h,0,w
    coords = [top,left,btm,right]

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid[0].shape, ps, stride0, dil)
    oh1,ow1,_,_ = comp_pads(vid[0].shape, ps, stride1, dil)
    n_h = (hp - (ps-1)*dil - 1)//stride0 + 1
    n_w = (wp - (ps-1)*dil - 1)//stride0 + 1

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h * n_w
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- init xsearch --
    prod_search = dnls.search.init("prod_with_index",flows.fflow, flows.bflow,
                                   k, ps, pt, ws, wt, oh0, ow0, oh1, ow1,
                                   dilation=dil, stride0=stride0, stride1=stride1,
                                   reflect_bounds=reflect_bounds,
                                   use_k=use_k,search_abs=search_abs,exact=exact)

    # -- init our inner product --
    h_off,w_off = oh1,ow1
    if not(use_unfold): h_off,w_off = 0,0
    adj,h_off,w_off = 0,0,0
    wpsum = dnls.reducers.WeightedPatchSumHeads(ps, pt, h_off=h_off, w_off=w_off,
                                                dilation=dil,
                                                reflect_bounds=reflect_bounds,
                                                adj=adj, exact=exact)

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


    #
    # -- forward pass --
    #

    # -- forward test --
    nq = nbatch
    qindex = 0
    scores,inds = prod_search(vid0_te,qindex,nbatch,vid1=vid1_te)
    scores_s = scores_to_heads_batch(scores,nheads,seed=123)
    wpatches_te = wpsum(vid2_te,scores_s,inds).view(b,nheads,nq,-1)

    # -- forward gt --
    scores,inds = prod_search(vid0_gt,qindex,nbatch,vid1=vid1_gt)
    scores_s = scores_to_heads_batch(scores,nheads,seed=123)
    wpatches_gt = simple_run_batch(vid2_gt,scores_s,inds,ps,pt,reflect_bounds,exact)

    # -- viz --
    # print(wpatches_te[:3,:3])
    # print(wpatches_gt[:3,:3])


    # -- confirm fwd --
    tol = 1e-6
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
        # print(grad_te[args][:10])
        # print(grad_gt[args][:10])
        # print(grad_te[args][:5])
        # print(grad_gt[args][:5])
        # print(diff.mean().item())
        # print(diff.max().item())
        Z = 1e-5 if idx > 0 else 1e-2
        # diff /= diff.max()
        # diff /= Z
        # if idx ==0: diff = diff.clamp(0.,1.)
        # dnls.testing.data.save_burst(diff,SAVE_DIR,"grad_diff_%d_%d"%(use_unfold,idx))

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

    th.cuda.synchronize()
