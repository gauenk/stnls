

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
    test_lists = {"ps":[7],"stride":[4],"dilation":[1],"wt":[0],
                  "ws":[-1,8],"top":[0],"btm":[64],"left":[0],"right":[64],"k":[-1,5],
                  "exact":[True],"seed":[123]}
    # test_lists = {"ps":[3,4,5,6,7,8],"stride":[1,2,3,4,5,8],"dilation":[1,2,3,4,5,8],
    #               "top":[1,11],"btm":[50,57],"left":[3,7],"right":[57,30]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def test_cu_vs_th_fwd(ps,stride,dilation,exact):
    """

    Test the CUDA code with torch code

    Forward Pass

    """


    # -- get args --
    dil = dilation
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    k,pt = 1,1
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
    vid = vid.to(device)[:,:1,].contiguous()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-2)
    vid = th.cat([vid,vid],-2)

    # -- normalize --
    vid /= vid.max()

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
    search = dnls.search.init("prod_with_index",flows.fflow, flows.bflow,
                              k, ps, pt, ws, wt, oh0, ow0, oh1, ow1,
                              chnls=-1,dilation=dil,
                              stride0=stride0, stride1=stride1,
                              reflect_bounds=reflect_bounds,use_k=False,
                              search_abs=True,use_adj=use_adj,
                              exact=exact)
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

    # -- flip cu --
    # print(score_te.shape) # == B,Q,K
    # score_te = rearrange(score_te,'(sh sw) (h w) -> h w sh sw',sh=n_h0,h=h)

    # -- run search --
    B = vid.shape[0]
    mode = "reflect" if reflect_bounds else "zero"
    score_gt = []
    for b in range(B):
        score_gt_b,_ = dnls.simple.prod_search_nn.run_nn(vid[b],ps,stride=stride0,
                                                         mode=mode,dilation=dil,
                                                         vid1=vidr[b])
        score_gt.append(score_gt_b)
    score_gt = th.stack(score_gt)
    score_gt = rearrange(score_gt,'b h w sh sw -> b (sh sw) (h w)')

    # -- viz --
    # print(score_gt.shape)
    # print(score_te.shape)

    # -- viz --
    # print(score_te[0,0,:5,:5])
    # print(score_gt[0,0,:5,:5])
    # print("-"*10)
    # print(score_te[0,0,5:10,5:10])
    # print(score_gt[0,0,5:10,5:10])
    # print("-"*10)
    # print(score_te[0,0,16:18,16:18])
    # print(score_gt[0,0,16:18,16:18])

    # diff = th.abs(score_te - score_gt).mean((-1,-2))
    # if diff.max() > 1e-5: diff /= diff.max()
    # diff = repeat(diff,'h w -> 1 c h w',c=3)
    # dnls.testing.data.save_burst(diff,SAVE_DIR,"nn2_diff")

    # diff = th.abs(score_te - score_gt).mean((0,1))
    # if diff.max() > 1e-5: diff /= diff.max()
    # diff = repeat(diff,'h w -> 1 c h w',c=3)
    # dnls.testing.data.save_burst(diff,SAVE_DIR,"nn2_diff_t")

    # -- compare --
    args0 = th.where(th.logical_not(th.isinf(score_gt))) # remove all inf
    diff = th.abs(score_te - score_gt) / (score_gt.abs() + 1e-5)
    diff = diff[args0]

    tol = 1e-5
    error = th.mean(th.abs(score_te - score_gt)).item()
    if error > tol: print("error: ",error)
    assert error < tol

    tol = 1e-4
    max_error = th.abs(score_te - score_gt).max().item()
    if max_error > tol: print("max error: ",max_error)
    assert max_error < tol


@pytest.mark.slow
def test_cu_vs_th_vid_bwd(ps,stride,dilation,exact):
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

def test_anchor_self(ps,k,ws,stride,dilation,exact):
    """

    Test the CUDA code with torch code

    Forward Pass

    """


    # -- get args --
    dil = dilation
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    pt = 1
    wt = 0
    # ws = -1
    # k = -1
    stride0 = stride
    stride1 = stride
    use_k = not(k > 0)
    search_abs = ws == -1
    assert stride1 == stride0 # searching with unfold needs stride1

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    reflect_bounds = True
    anchor_self = True
    use_self = True
    full_ws = False
    use_adj = False
    if use_k is False: return # skip not "topk"

    # -- load data --
    vid = dnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid.to(device)[:,:1,].contiguous()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-2)
    vid = th.cat([vid,vid],-2)

    # -- normalize --
    vid /= vid.max()

    # -- compute flow --
    flows = dnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape
    chnls = vid.shape[-3]

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid[0].shape, ps, stride0, dil)
    oh1,ow1,_,_ = comp_pads(vid[0].shape, ps, stride1, dil)
    _,_,n0,n1 = get_batching_info(vid[0].shape,stride0,stride1,ps,dil)
    n_h0,n_w0 = n0[0],n0[1]
    n_h1,n_w1 = n1[0],n1[1]
    oh0, ow0, oh1, ow1 = 0, 0, 0, 0

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h0 * n_w0
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- exec fold fxns --
    search = dnls.search.init("prod_with_index",flows.fflow, flows.bflow,
                              k, ps, pt, ws, wt, oh0, ow0, oh1, ow1,
                              chnls=-1,dilation=dil,
                              stride0=stride0, stride1=stride1,
                              reflect_bounds=reflect_bounds,use_k=use_k,
                              search_abs=search_abs,use_adj=use_adj,
                              full_ws = full_ws,
                              anchor_self=anchor_self,use_self=use_self,
                              exact=exact)

    # -- run search --
    vidr = th.rand_like(vid)
    qindex = 0
    dists_te,inds_te = search(vid,qindex,nbatch,vid1=vidr)
    dists_te = dists_te[:,:,0] # only self
    inds_te = inds_te[:,:,0] # only self

    # -- [gt dists,inds] compute first prod --
    dists_gt = compute_gt_dists(vid,vidr,ps,stride0,dil,reflect_bounds)
    inds_gt = compute_gt_inds(vid.shape,stride0,inds_te.device)

    # -- compare inds --
    diff = th.sum(th.abs(inds_gt - inds_te)).item()
    assert diff < 1e-10

    # -- compare --
    args0 = th.where(th.logical_not(th.isinf(dists_gt))) # remove all inf
    diff = th.abs(dists_te - dists_gt) / (dists_gt.abs() + 1e-5)
    diff = diff[args0]

    tol = 1e-5
    error = th.mean(th.abs(dists_te - dists_gt)).item()
    if error > tol: print("error: ",error)
    assert error < tol

    tol = 1e-4
    max_error = th.abs(dists_te - dists_gt).max().item()
    if max_error > tol: print("max error: ",max_error)
    assert max_error < tol


def test_cu_vs_th_params_bwd(ps,stride,dilation,exact,seed):
    """

    Test the CUDA code with torch code

    Backward Pass for parameters for video

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
    set_seed(seed)

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
    vid = th.cat([vid,vid],-3)
    vid = th.cat([vid,vid],-3)
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

    # -- exec fold fxns --
    use_adj = True
    # use_adj = False
    # oh0, ow0, oh1, ow1 = 0,0,0,0
    search = dnls.search.init("prod_with_index",flows.fflow, flows.bflow, k, ps, pt,
                              ws, wt, oh0, ow0, oh1, ow1,use_adj=use_adj,
                              chnls=-1,dilation=dil,
                              stride0=stride0,stride1=stride1,
                              reflect_bounds=reflect_bounds,
                              use_k=False,exact=exact,search_abs=True)
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

    # -- declare weights --
    def create_weights(ichnls,ochnls):
        gam = th.nn.Conv2d(in_channels=ichnls, out_channels=ochnls,
                        kernel_size=1, stride=1,padding=0)
        phi = th.nn.Conv2d(in_channels=ichnls, out_channels=ochnls,
                        kernel_size=1, stride=1,padding=0)
        # theta = th.nn.Conv2d(in_channels=ichnls, out_channels=ochnls,
        #                 kernel_size=1, stride=1,padding=0)
        return gam,phi

    def create_weights_pair(ichnls,ochnls):
        params0 = create_weights(ichnls,ochnls)
        params1 = create_weights(ichnls,ochnls)
        for param0,param1 in zip(params0,params1):
            param0.weight.data = th.randn_like(param0.weight.data)
            param1.weight.data.copy_(param0.weight.data)
            param0.bias.data = th.randn_like(param0.bias.data)
            param1.bias.data.copy_(param0.bias.data)
        return params0,params1

    def get_xformed(vid,params):
        B,T = vid.shape[:2]
        xformed = []
        vid = rearrange(vid,'b t c h w -> (b t) c h w')
        for param in params:
            param.requires_grad_(True)
            param = param.to(vid.device)
            _xform = param(vid)
            # _xform = _xform.abs()
            _xform = _xform.clip(-1.,1.)
            # _xform = th.round(_xform,decimals=2).type(th.float)
            _xform = rearrange(_xform,'(b t) c h w -> b t c h w',b=B)
            xformed.append(_xform)
        # xformed = [vid,vid]
        return xformed

    # -- allow grads --
    ichnls,ochnls = vid.shape[2],16
    params_te,params_gt = create_weights_pair(ichnls,ochnls)
    vid0_te,vid1_te = get_xformed(vid.clone(),params_te)
    vid0_gt,vid1_gt = get_xformed(vid.clone(),params_gt)
    # print(vid0_te.mean(),vid0_te.std())

    # -- check fwd with weights --
    error0 = th.sum(th.abs(vid0_te - vid0_gt)).item()
    error1 = th.sum(th.abs(vid1_te - vid1_gt)).item()
    assert error0 < 1e-10
    assert error1 < 1e-10
    # print("errors: ",error0,error1)

    #
    # -- run search --
    #

    # -- run cu --
    qindex = 0
    score_te,inds_te = search(vid0_te,qindex,nbatch,vid1=vid1_te)
    # score_te = rearrange(score_te,'b (sh sw) (h w) -> b h w sh sw',sh=n_h0,h=h)
    # score_te = rearrange(score_te,'b (sh sw) (h w) -> b h w sh sw',sh=n_h0,h=h)

    # -- run nn --
    mode = "reflect" if reflect_bounds else "zero"
    score_gt = dnls.simple.prod_search_nn.run_nn_batch(vid0_gt,ps,stride=stride0,
                                                         dilation=dil,
                                                         vid1=vid1_gt,mode=mode)
    score_gt = score_gt.view(b,h*w,-1).transpose(1,2)

    # -- vis --
    # diff = th.abs(score_te - score_gt)/(score_gt.abs()+1e-10)
    # args = th.where(diff>1e-10)
    # save_burst = dnls.testing.data.save_burst
    # for i in range(len(args)):
    #     print(i,th.unique(args[i]))
    # if diff.max() > 1e-10: diff /= diff.max()
    # save_burst(diff[0,0][None,None],"./output/tests/prod_search/","diff")
    # save_burst(diff[:,:,0,0][None,None],"./output/tests/prod_search/","diff_d00")

    # -- compare fwd --
    diff = th.abs(score_te - score_gt)
    args = th.where(diff > 1e-3)
    # print(score_te[args][:5])
    # print(score_gt[args][:5])
    max_error = th.abs(score_te - score_gt).max().item()
    # print("max error: ",max_error)
    assert max_error < 1e-3

    error = th.mean(th.abs(score_te - score_gt)).item()
    # print("error: ",error)
    assert error < 1e-4

    # -- compute grad --
    score_grad = 2.*(th.rand_like(score_gt)-0.5)
    th.autograd.backward(score_gt,score_grad)
    th.autograd.backward(score_te,score_grad)

    # -- for both grads --
    _grads_te = [p.weight.grad for p in params_te]
    _grads_gt = [p.weight.grad for p in params_gt]
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


def test_cu_vs_simp_fwd(ps,stride,dilation,top,btm,left,right,k,exact):


    # -- get args --
    dil = dilation
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    chnls,pt,wt = 3,1,0
    ws = -1 if k == -1 else 10
    search_abs = k == -1
    use_k = not(k == -1)
    # print(ws,k,search_abs,use_k)
    use_adj = True#k==-1

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    adj = 0

    # -- load data --
    vid = dnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid.to(device)[:,:2,].contiguous()/255.
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- compute flow --
    flows = dnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)
    flows.fflow = 10*th.randn_like(flows.fflow)
    flows.bflow = 10*th.randn_like(flows.fflow)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape

    # -- batching info --
    stride0 = stride
    stride1 = 1
    npix = t * h * w
    nh = (h-1)//stride0+1
    nw = (w-1)//stride0+1
    ntotal = t * nh * nw
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- pads --
    oh0,ow0,_,_ = comp_pads(vid[0].shape, ps, stride0, dil)
    oh1,ow1,hp,wp = comp_pads(vid[0].shape, ps, stride1, dil)

    # -- exec fold fxns --
    search = dnls.search.init("prod_with_index",flows.fflow, flows.bflow, k, ps, pt,
                              ws, wt, oh0, ow0, oh1, ow1,
                              chnls=chnls,dilation=dil,
                              stride0=stride0, stride1=stride1,
                              use_k=use_k,search_abs=search_abs,
                              reflect_bounds=True,exact=True,use_adj=use_adj)
    fold_nl = dnls.iFold(vshape,None,stride=stride1,dilation=dil,adj=adj)
    patches_nl = []
    gpu_mem.print_gpu_stats(gpu_stats,"start-exec")

    # -- query inds
    coords = [0,t,0,0,h,w]
    qindex = 0
    iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch,stride0,
                                                coords,t,device)

    # -- run search --
    score_te,inds_te = search(vid,qindex,nbatch)
    score_simp,inds_simp = dnls.simple.prod_search.run(vid,iqueries,flows,k,
                                                       ps,pt,ws,wt,chnls,
                                                       stride0=stride0,stride1=stride1,
                                                       dilation=dil,use_k=use_k,
                                                       use_bound=True,use_adj=use_adj,
                                                       search_abs=search_abs)

    # -- reshape --
    nq = iqueries.shape[0]
    score_te = score_te.view(nq,-1)
    score_simp = score_simp.view(nq,-1)

    # -- viz --
    # diff2 = th.abs(score_te - score_simp)
    # nh = 32#np.sqrt(diff2.shape[0])
    # print(diff2.shape)
    # diff2 = rearrange(diff2,'(h w) k -> k 1 h w',h=nh)
    # print(diff2.shape)
    # dnls.testing.data.save_burst(diff2,SAVE_DIR,"diff")
    # print(score_te[:3,:])
    # print(score_simp[:3,:])

    # -- compare --
    error = th.mean(th.abs(score_te - score_simp)).item()
    assert error < 1e-6

# @pytest.mark.skip(reason="too long right now")
def test_batched(ps,stride,dilation,top,btm,left,right,ws,wt):

    # -- get args --
    dil = dilation
    ext = "jpg"
    dnames = ["davis_baseball_64x64","davis_baseball_64x64"]
    k,pt = 1,1
    ws,wt = 10,0
    ws = -1
    k = -1

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    exact = True
    gpu_stats = False
    use_adj = True
    reflect_bounds = False

    # -- load data --
    vid = dnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid[:,:1].to(device).contiguous()
    flows = dnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    b,t,color,h,w = shape
    vshape = vid.shape

    # -- allow grads --
    vid_te = vid.clone()
    vid_gt = vid.clone()
    vidr_te = vid_te.clone()
    vidr_gt = vid_gt.clone()
    vid_te.requires_grad_(True)
    vid_gt.requires_grad_(True)
    vidr_te.requires_grad_(True)
    vidr_gt.requires_grad_(True)

    # -- batching info --
    stride0 = stride
    stride1 = 1
    npix = t * h * w
    nh = (h-1)//stride0+1
    nw = (w-1)//stride0+1
    ntotal = t * nh * nw
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid[0].shape, ps, stride0, dil)
    oh1,ow1,_,_ = comp_pads(vid[0].shape, ps, stride1, dil)

    # -- exec fold fxns --
    search = dnls.search.init("prod_with_index",
                              flows.fflow, flows.bflow, k, ps, pt,
                              ws, wt, oh0, ow0, oh1, ow1,use_adj=use_adj,
                              chnls=-1,dilation=dil,
                              stride0=stride0, stride1=stride1,
                              reflect_bounds=reflect_bounds,
                              use_k=False,exact=exact,search_abs=True)

    # -- run prod_search over batches --
    score_te = []
    for index in range(nbatches):

        # -- batch info --
        qindex = min(nbatch * index,npix)
        nbatch_i =  min(nbatch, ntotal - qindex)

        # -- run prod_search --
        score_te_i,inds_te = search(vid_te,qindex,nbatch_i,vid1=vidr_te)
        score_te.append(score_te_i)

    # -- forward reference --
    mode = "reflect" if reflect_bounds else "zero"
    score_gt = dnls.simple.prod_search_nn.run_nn_batch(vid_gt,ps,
                                                       stride=stride0,mode=mode,
                                                       dilation=dil,vid1=vidr_gt)
    score_gt = score_gt.view(b,h*w,-1).transpose(2,1)

    # -- compare forward --
    score_te_cat = th.cat(score_te,0)
    error = th.abs(score_gt - score_te_cat).mean()
    assert error < 1e-7
    error = th.abs(score_gt - score_te_cat).max()
    assert error < 1e-6

    # -- run backward --
    vid_grad = th.randn_like(vid_gt)
    th.autograd.backward(vid_gt,vid_grad)
    th.autograd.backward(vid_te,vid_grad)
    # gpu_mem.print_gpu_stats(gpu_stats,"post-bkw")
    # dnls.testing.data.save_burst(vid_gt,"./output/","vid_gt")
    # dnls.testing.data.save_burst(vid_nl,"./output/","vid_nl")

    # -- get grads --
    grad_gt = vid_gt.grad
    grad_te = vid_te.grad

    # -- check backward --
    error = th.sum((grad_gt - grad_te)**2).item()
    assert error < 1e-10

    # -- clean-up --
    th.cuda.empty_cache()
    del vid_te,vidr_te
    del vid_gt,vidr_gt
    del grad_gt,grad_te
    del score_te,score_te_cat,score_gt
    th.cuda.empty_cache()
    th.cuda.synchronize()

def compute_gt_inds(vshape,stride0,device):

    # -- create mesh --
    B,T,C,H,W = vshape
    nH = (H-1)//stride0+1
    nW = (W-1)//stride0+1
    hgrid = np.arange(nH)*stride0
    wgrid = np.arange(nW)*stride0
    lmesh = np.meshgrid(*[hgrid,wgrid])
    mesh = [grid.ravel() for grid in lmesh]
    mesh = np.flip(np.stack(mesh,-1),-1)
    mesh = mesh.copy()
    # print("mesh.shape: ",mesh.shape)

    # -- torchify --
    inds_hw = th.from_numpy(mesh).to(device)
    inds_hw = inds_hw.type(th.int32)
    N = inds_hw.shape[0]

    # -- add "t" --
    inds = []
    for t in range(T):
        inds_t = t*th.ones((N,1),device=device,dtype=th.int32)
        inds_t = th.cat([inds_t,inds_hw],-1)
        inds.append(inds_t)
    inds = th.stack(inds)

    # -- repeat across "b" --
    inds = repeat(inds,'t n thr -> b t n thr',b=B)
    inds = inds.reshape(B,T*N,3)
    return inds


def compute_gt_dists(vid,vidr,ps,stride0,dil,rmode):
    B = vid.shape[0]
    dists_gt = []
    for b in range(B):
        patches = run_unfold(vid[b],ps,stride0,dil,rmode)
        patches_r = run_unfold(vidr[b],ps,stride0,dil,rmode)
        dists_gt_b = th.sum(patches * patches_r,dim=(-1))
        dists_gt.append(dists_gt_b)
    dists_gt = th.stack(dists_gt)
    return dists_gt

def run_unfold(_vid,_ps,_stride=1,_dil=1,rmode=False):

    # -- avoid fixutres --
    vid,stride = _vid,_stride
    ps,dil = _ps,_dil
    pads = (ps//2,)*4
    mode = "reflect" if rmode else "constant"

    # -- run --
    shape_str = 't d hw -> (t hw) d'
    vid_pad = pad(vid,pads,mode=mode)
    patches = unfold(vid_pad,(ps,ps),stride=stride,dilation=dil)
    patches = rearrange(patches,shape_str)
    return patches
