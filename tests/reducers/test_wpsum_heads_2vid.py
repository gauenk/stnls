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
from dnls.utils.inds import get_nums_hw
from dnls.utils.pads import same_padding,comp_pads

# -- test func --
from torch.nn.functional import fold,unfold,pad,softmax,log_softmax
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/wpsum_heads_2vid/")

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

def scores_to_heads(scores,nheads,seed=None):
    if not(seed is None):
        th.manual_seed(seed)
    score_heads = []
    for head in range(nheads):
        scores_h = softmax(scores*10  + th.randn_like(scores) ,1)
        score_heads.append(scores_h)
    score_heads = th.stack(score_heads,-1) # a b nheads
    return score_heads

def simple_run(vid,score_heads,inds,ps,pt,reflect_bounds,exact):
    unfold_k = dnls.UnfoldK(ps,pt,exact=exact,reflect_bounds=reflect_bounds)
    patches_i = unfold_k(vid,inds).type(th.float64)
    patches_i = rearrange(patches_i,'n k 1 c h w -> n k 1 (c h w)')
    wpatches_i = th.sum(score_heads[...,None] * patches_i,1).type(th.float32)
    return wpatches_i

@pytest.mark.skip("not implemented.")
def test_forward(ps,stride,dilation,top,btm,left,right,k,exact):

    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
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
    flows = dnls.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

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
    oh1,ow1,hp1,wp1 = comp_pads(vid.shape, ps, stride1, dil)
    n_h0,n_w0 = get_nums_hw(vid.shape,stride0,ps,dil,"pad_same")

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h0 * n_w0
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- init xsearch --
    search = dnls.search.init("prod_with_index",flows.fflow, flows.bflow,
                              k, ps, pt, ws, wt, oh0, ow0, oh1, ow1,
                              dilation=dil, stride0=stride0, stride1=stride1,
                              use_k=use_k,reflect_bounds=reflect_bounds,
                              search_abs=search_abs)

    # -- init our inner product --
    t,c,h,w = vshape
    vshape = (t,c*nheads,h,w)
    h_off,w_off = oh1,ow1
    if not(use_unfold): h_off,w_off = 0,0
    adj,h_off,w_off = 0,0,0
    fold = dnls.iFold(vshape,coords,stride=stride0,dilation=dil,
                      adj=ps//2,use_reflect=True,only_full=True)
    wfold = dnls.iFold(vshape,coords,stride=stride0,dilation=dil,
                       adj=ps//2,use_reflect=True,only_full=True)
    wpsum = dnls.reducers.WeightedPatchSumHeads(ps, pt, h_off=h_off,w_off=w_off,
                                                dilation=dil,
                                                reflect_bounds=reflect_bounds,
                                                adj=adj, exact=exact)
    wpsum2vid = dnls.reducers.WeightedPatchSumHeads2Vid(ps, pt,
                                                        h_off=h_off,w_off=w_off,
                                                        stride=stride0,dilation=dil,
                                                        reflect_bounds=reflect_bounds,
                                                        adj=adj, exact=exact)


    # -- compute scores --
    qindex = 0
    scores,inds = search(vid,qindex,nbatch,vid1=vid)
    scores_s = scores_to_heads(scores,nheads)

    # -- baseline --
    wpatches_te = wpsum(vid,scores_s,inds)
    wpatches_te = rearrange(wpatches_te,'b k c h w -> b 1 1 (k c) h w')
    ones = th.ones_like(wpatches_te)
    print(wpatches_te.shape)
    fold(wpatches_te,0)
    wfold(ones,0)
    folded_gt = fold.vid / (wfold.vid + 1e-5)
    print(folded_gt.shape)
    folded_gt = rearrange(folded_gt,'b (k c) h w -> b k c h w',k=nheads)
    print("folded_gt.shape: ",folded_gt.shape)

    # -- our method --
    Z = rearrange(wfold.vid,'b (k c) h w -> b k c h w',k=nheads)
    folded_te = wpsum2vid(vid,scores_s,inds,qindex)  / (Z + 1e-5)
    print("folded_te.shape: ",folded_te.shape)

    # -- compare --
    diff = th.abs(folded_gt - folded_te) / (folded_gt.abs() + 1e-5)

    tol = 1e-7
    error = diff.mean().item()
    if error > tol: print(error)
    assert error < tol

    tol = 1e-6
    error = diff.max().item()
    if error > tol: print(error)
    assert error < tol

@pytest.mark.skip("not implemented.")
def test_backward(ps,stride,dilation,top,btm,left,right,k):

    # -- get args --
    pt,dil = 1,dilation
    dname,ext = "davis_baseball_64x64","jpg"
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
    flows = dnls.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

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
    prod_search = dnls.search.init("prod",flows.fflow, flows.bflow, k, ps, pt,
                               ws, wt, oh0, ow0, oh1, ow1,
                               dilation=dil, stride=stride1,
                               reflect_bounds=reflect_bounds,
                               use_k=use_k,search_abs=search_abs,exact=exact)

    # -- query inds --
    qindex = 0
    iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch,stride0,
                                                coords,t,device)

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
    nq = iqueries.shape[0]
    scores,inds = prod_search(vid0_te,iqueries,vid1=vid1_te)
    scores_s = scores_to_heads(scores,nheads,seed=123)
    wpatches_te = wpsum(vid2_te,scores_s,inds).view(nq,nheads,-1)

    # -- forward gt --
    scores,inds = prod_search(vid0_gt,iqueries,vid1=vid1_gt)
    scores_s = scores_to_heads(scores,nheads,seed=123)
    wpatches_gt = simple_run(vid2_gt,scores_s,inds,ps,pt,reflect_bounds,exact)

    # -- viz --
    # print(wpatches_te[:3,:3])
    # print(wpatches_gt[:3,:3])


    # -- confirm fwd --
    tol = 1e-7
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

