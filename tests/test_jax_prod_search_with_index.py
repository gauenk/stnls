
# -- data mgnmt --
from pathlib import Path

# -- testing --
import pytest

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- jax --
import jax
from jax._src import api
import jax.numpy as jnp
import jax.random as jr
from functools import partial

# -- dnls --
import dnls
import dnls.utils.gpu_mem as gpu_mem
from dnls.utils.pads import comp_pads
from dnls.utils.inds import get_batching_info

# -- paths --
SAVE_DIR = Path("./output/tests/prod_search")

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
                  "ws":[-1],"top":[0],"btm":[64],"left":[0],"right":[64],"k":[-1,5],
                  "exact":[True]}
    # test_lists = {"ps":[3,4,5,6,7,8],"stride":[1,2,3,4,5,8],"dilation":[1,2,3,4,5,8],
    #               "top":[1,11],"btm":[50,57],"left":[3,7],"right":[57,30]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)


@pytest.mark.jax
def test_cu_vs_th_fwd(ps,stride,dilation,exact):
    """

    Test the CUDA code with torch code

    Forward Pass

    """

    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    k,pt = 1,1
    wt = 0
    ws = 8
    k = 4
    stride0 = stride
    stride1 = 1
    search_abs = ws <= 0
    use_k = k > 0

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    adj = False
    reflect_bounds = False
    use_adj = True

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

    # -- compute flow --
    flows = dnls.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)
    flows_jax = dnls.flow.pth2jax(flows)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    nframes,chnls = vid.shape[:2]
    vshape = vid.shape

    # -- sub square --
    top,btm,left,right = 0,h,0,w
    coords = [top,left,btm,right]
    # sq_h = coords[2] - coords[0]
    # sq_w = coords[3] - coords[1]

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid.shape, ps, stride0, dil)
    oh1,ow1,_,_ = comp_pads(vid.shape, ps, stride1, dil)
    _,_,n0,n1 = get_batching_info(vid.shape,stride0,stride1,ps,dil)
    n_h0,n_w0 = n0[0],n0[1]
    n_h1,n_w1 = n1[0],n1[1]

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h0 * n_w0
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- two video types --
    vid_pth = vid.clone()
    vid_jax = jnp.array(vid.clone().cpu().numpy())

    # -- random videos --
    vidr_pth = th.rand_like(vid_pth)
    vidr_jax = jnp.array(vidr_pth.clone().cpu().numpy())

    # -- exec fold fxns --
    # oh0, ow0, oh1, ow1 = 0, 0, 0, 0
    # oh0, ow0, oh1, ow1 = -oh0, -ow0, -oh1, -ow1
    search_gt = dnls.search.init("prod_with_index",flows.fflow, flows.bflow,
                                 k, ps, pt, ws, wt, oh0, ow0, oh1, ow1,
                                 chnls=-1,dilation=dil,
                                 stride0=stride0, stride1=stride1,
                                 reflect_bounds=reflect_bounds,use_k=use_k,
                                 search_abs=search_abs,use_adj=use_adj,
                                 exact=exact)
    # chnls=-1, dilation=1, stride0=1, stride1=1,
    # use_k=True, use_adj=True, reflect_bounds=True,
    # search_abs=False, full_ws = False, nbwd=1, exact=False,
    # h0_off=0,w0_off=0,h1_off=0,w1_off=0,remove_self=False,
    # anchor_self=False,rbwd=True):
    search_te = dnls.jax.search.init("prod_with_index",
                                     flows_jax.fflow, flows_jax.bflow,
                                     nframes, k, ps, pt, ws, wt,
                                     oh0, ow0, oh1, ow1,
                                     chnls=-1,dilation=dil,
                                     stride0=stride0, stride1=stride1,
                                     reflect_bounds=reflect_bounds,use_k=use_k,
                                     search_abs=search_abs,use_adj=use_adj,
                                     exact=exact)
    search_te = jax.jit(search_te,static_argnums=(1,2))

    # print(search_gt)
    # print(search_te)
    # exit(0)
    # -- run pytorch search --
    qindex = 0
    score_gt,inds_gt = search_gt(vid_pth,qindex,nbatch,vid1=vidr_pth)

    # -- run jax search --
    score_te,inds_te = search_te(vid_jax,qindex,nbatch,vid1=vidr_jax)
    score_te = th.from_numpy(np.asarray(score_te)).to(device)
    inds_te = th.from_numpy(np.asarray(inds_te)).to(device)

    # -- viz --
    # print(score_gt[:3,:3])
    # print(score_te[:3,:3])
    # # print(score_te[:3,-3:])
    # print("-"*50)
    # print(score_gt[-3:,:3])
    # print(score_te[-3:,:3])
    # print("-"*50)
    # print(inds_gt[:3,:3])
    # print(inds_te[:3,:3])
    # print("-"*50)
    # print(inds_gt[32:35,:3])
    # print(inds_te[32:35,:3])
    # print("-"*50)
    # print(inds_gt[-3:,:3])
    # print(inds_te[-3:,:3])
    # print("-"*50)

    # print(inds_te[:3,-3:])
    # args = th.where(th.abs(score_gt - score_te)>0.1)
    # print(score_gt[args])
    # print(score_te[args])
    # print(args)
    # print(th.unique(args[0]))

    # -- compare --
    args0 = th.where(th.logical_not(th.isinf(score_gt))) # remove all inf
    args1 = th.where(th.logical_not(th.isinf(score_te))) # remove all inf
    diff = th.abs(score_te - score_gt) / (score_gt.abs() + 1e-5)
    diff = diff[args0]
    # print(th.abs(args0[0] - args1[0]).sum())
    # print(th.abs(args0[1] - args1[1]).sum())

    tol = 1e-5
    error = diff.mean().item()
    if error > tol: print("error: ",error)
    assert error < tol

    tol = 1e-4
    max_error = diff.max().item()
    if max_error > tol: print("max error: ",max_error)
    assert max_error < tol


@pytest.mark.jax
def test_cu_vs_th_bwd(ps,stride,dilation,exact):
    """

    Test the CUDA code with torch code

    Backward Pass

    """

    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    k,pt = 1,1
    wt = 0
    ws = 8
    k = 4
    stride0 = stride
    stride1 = 1
    search_abs = ws <= 0
    use_k = k > 0

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    adj = False
    reflect_bounds = False
    use_adj = True

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

    # -- compute flow --
    flows = dnls.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)
    flows_jax = dnls.flow.pth2jax(flows)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    nframes,chnls = vid.shape[:2]
    vshape = vid.shape

    # -- sub square --
    top,btm,left,right = 0,h,0,w
    coords = [top,left,btm,right]
    # sq_h = coords[2] - coords[0]
    # sq_w = coords[3] - coords[1]

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid.shape, ps, stride0, dil)
    oh1,ow1,_,_ = comp_pads(vid.shape, ps, stride1, dil)
    _,_,n0,n1 = get_batching_info(vid.shape,stride0,stride1,ps,dil)
    n_h0,n_w0 = n0[0],n0[1]
    n_h1,n_w1 = n1[0],n1[1]

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h0 * n_w0
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- two video types --
    vid_pth = vid.clone()
    vid_jax = jnp.array(vid.clone().cpu().numpy())

    # -- random videos --
    vidr_pth = th.rand_like(vid_pth)
    vidr_jax = jnp.array(vidr_pth.clone().cpu().numpy())
    qindex = 0

    # -- exec fold fxns --
    # oh0, ow0, oh1, ow1 = 0, 0, 0, 0
    # oh0, ow0, oh1, ow1 = -oh0, -ow0, -oh1, -ow1
    search_gt = dnls.search.init("prod_with_index",flows.fflow, flows.bflow,
                                 k, ps, pt, ws, wt, oh0, ow0, oh1, ow1,
                                 chnls=-1,dilation=dil,
                                 stride0=stride0, stride1=stride1,
                                 reflect_bounds=reflect_bounds,use_k=use_k,
                                 search_abs=search_abs,use_adj=use_adj,
                                 exact=exact)
    # chnls=-1, dilation=1, stride0=1, stride1=1,
    # use_k=True, use_adj=True, reflect_bounds=True,
    # search_abs=False, full_ws = False, nbwd=1, exact=False,
    # h0_off=0,w0_off=0,h1_off=0,w1_off=0,remove_self=False,
    # anchor_self=False,rbwd=True):
    search_te = dnls.jax.search.init("prod_with_index",
                                     flows_jax.fflow, flows_jax.bflow,
                                     nframes, k, ps, pt, ws, wt,
                                     oh0, ow0, oh1, ow1,
                                     chnls=-1,dilation=dil,
                                     stride0=stride0, stride1=stride1,
                                     reflect_bounds=reflect_bounds,use_k=use_k,
                                     search_abs=search_abs,use_adj=use_adj,
                                     exact=exact)
    # search_fwd = jax.jit(search_te,static_argnums=(1,2))
    search_fwd = search_te


    # print(search_gt)
    # print(search_te)
    # exit(0)
    # -- run pytorch search --
    score_gt,inds_gt = search_gt(vid_pth,qindex,nbatch,vid1=vidr_pth)

    # -- create gradients --
    score_grad_pth = th.randn_like(score_gt)
    score_grad_jax = jnp.array(score_grad_pth.cpu().numpy())

    # -- run jax search --
    score_te,inds_te = search_fwd(vid_jax,qindex,nbatch,vidr_jax)
    inds_grad = jnp.array(th.zeros_like(inds_gt).cpu().numpy())
    # grad_in = (score_grad_jax,)
    grad_in = (vid_jax,qindex,nbatch,vidr_jax)
    grad_out = (score_te,None,None,None)

    # grad_in = (vid_jax,qindex,nbatch,vidr_jax) # should be this.
    grad_in = (vid_jax,vidr_jax)
    grad_out = (score_grad_jax,inds_grad)
    search_p = lambda x,y: search_fwd(x,qindex,nbatch,y)
    og,vjp_fxn = jax.vjp(search_p,*grad_in)#,grad_out,has_aux=True)
    # print(vjp_fxn)
    # print(len(og))
    print("outside.")
    out = vjp_fxn(grad_out)
    print(len(out))
    print(out[0].shape)
    print(out[1].shape)
    # a,b,c = jvp_out
    # print(len(jvp_out))
    # print(a.shape,b.shape,c.shape)

    exit(0)

    # -- compare --
    args0 = th.where(th.logical_not(th.isinf(score_gt))) # remove all inf
    args1 = th.where(th.logical_not(th.isinf(score_te))) # remove all inf
    diff = th.abs(score_te - score_gt) / (score_gt.abs() + 1e-5)
    diff = diff[args0]
    # print(th.abs(args0[0] - args1[0]).sum())
    # print(th.abs(args0[1] - args1[1]).sum())

    tol = 1e-5
    error = diff.mean().item()
    if error > tol: print("error: ",error)
    assert error < tol

    tol = 1e-4
    max_error = diff.max().item()
    if max_error > tol: print("max error: ",max_error)
    assert max_error < tol


@pytest.mark.jax
def test_cu_vs_th_jvp(ps,stride,dilation,exact):
    """

    Test the CUDA code with torch code

    Backward Pass

    """

    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    k,pt = 1,1
    wt = 0
    ws = 8
    k = 4
    stride0 = stride
    stride1 = 1
    search_abs = ws <= 0
    use_k = k > 0

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    adj = False
    reflect_bounds = False
    use_adj = True

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

    # -- compute flow --
    flows = dnls.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)
    flows_jax = dnls.flow.pth2jax(flows)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    nframes,chnls = vid.shape[:2]
    vshape = vid.shape

    # -- sub square --
    top,btm,left,right = 0,h,0,w
    coords = [top,left,btm,right]
    # sq_h = coords[2] - coords[0]
    # sq_w = coords[3] - coords[1]

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid.shape, ps, stride0, dil)
    oh1,ow1,_,_ = comp_pads(vid.shape, ps, stride1, dil)
    _,_,n0,n1 = get_batching_info(vid.shape,stride0,stride1,ps,dil)
    n_h0,n_w0 = n0[0],n0[1]
    n_h1,n_w1 = n1[0],n1[1]

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h0 * n_w0
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- two video types --
    vid_pth = vid.clone()
    vid_jax = jnp.array(vid.clone().cpu().numpy())

    # -- random videos --
    vidr_pth = th.rand_like(vid_pth)
    vidr_jax = jnp.array(vidr_pth.clone().cpu().numpy())
    qindex = 0

    # -- exec fold fxns --
    # oh0, ow0, oh1, ow1 = 0, 0, 0, 0
    # oh0, ow0, oh1, ow1 = -oh0, -ow0, -oh1, -ow1
    search_gt = dnls.search.init("prod_with_index",flows.fflow, flows.bflow,
                                 k, ps, pt, ws, wt, oh0, ow0, oh1, ow1,
                                 chnls=-1,dilation=dil,
                                 stride0=stride0, stride1=stride1,
                                 reflect_bounds=reflect_bounds,use_k=use_k,
                                 search_abs=search_abs,use_adj=use_adj,
                                 exact=exact)
    # chnls=-1, dilation=1, stride0=1, stride1=1,
    # use_k=True, use_adj=True, reflect_bounds=True,
    # search_abs=False, full_ws = False, nbwd=1, exact=False,
    # h0_off=0,w0_off=0,h1_off=0,w1_off=0,remove_self=False,
    # anchor_self=False,rbwd=True):
    search_te = dnls.jax.search.init("prod_with_index",
                                     flows_jax.fflow, flows_jax.bflow,
                                     nframes, k, ps, pt, ws, wt,
                                     oh0, ow0, oh1, ow1,
                                     chnls=-1,dilation=dil,
                                     stride0=stride0, stride1=stride1,
                                     reflect_bounds=reflect_bounds,use_k=use_k,
                                     search_abs=search_abs,use_adj=use_adj,
                                     exact=exact)
    # search_fwd = jax.jit(search_te,static_argnums=(1,2))
    search_fwd = search_te


    # print(search_gt)
    # print(search_te)
    # exit(0)
    # -- run pytorch search --
    score_gt,inds_gt = search_gt(vid_pth,qindex,nbatch,vid1=vidr_pth)

    # -- create gradients --
    score_grad_pth = th.randn_like(score_gt)
    score_grad_jax = jnp.array(score_grad_pth.cpu().numpy())

    # -- run jax search --
    score_te,inds_te = search_fwd(vid_jax,qindex,nbatch,vidr_jax)
    inds_grad = jnp.array(th.zeros_like(inds_gt).cpu().numpy())
    # grad_in = (score_grad_jax,)
    grad_in = (vid_jax,qindex,nbatch,vidr_jax)
    grad_out = (score_te,None,None,None)

    # grad_in = (vid_jax,qindex,nbatch,vidr_jax) # should be this.
    grad_in = (vid_jax,vidr_jax)
    grad_out = (vid_jax,vidr_jax)
    search_p = lambda x,y: search_fwd(x,qindex,nbatch,y)
    jvp_out = jax.jvp(search_p,grad_in,grad_out,has_aux=True)
    a,b,c = jvp_out
    print(len(jvp_out))
    print(a.shape,b.shape,c.shape)
    exit(0)

    # jvp_out = jax.jvp(search_fwd,grad_in,grad_out,has_aux=True)
    # search_jvp = jax.jit(api.jvp(search_te,has_aux),static_argnums=(1,2))
    # search_grad = jax.jit(api.grad(search_te,has_aux=True),static_argnums=(1,2))
    # search_jvp((score_te,inds_te),(vid_jax,vidr_jax))
    # score_te = th.from_numpy(np.asarray(score_te)).to(device)
    # inds_te = th.from_numpy(np.asarray(inds_te)).to(device)
    # print(type(qindex))
    # print(search_grad)
    # print(search_grad(vid_jax,qindex,nbatch,vidr_jax))
    # print(search_grad(vid_jax,vidr_jax))
    # exit(0)


    # -- viz --
    # print(score_gt[:3,:3])
    # print(score_te[:3,:3])
    # # print(score_te[:3,-3:])
    # print("-"*50)
    # print(score_gt[-3:,:3])
    # print(score_te[-3:,:3])
    # print("-"*50)
    # print(inds_gt[:3,:3])
    # print(inds_te[:3,:3])
    # print("-"*50)
    # print(inds_gt[32:35,:3])
    # print(inds_te[32:35,:3])
    # print("-"*50)
    # print(inds_gt[-3:,:3])
    # print(inds_te[-3:,:3])
    # print("-"*50)

    # print(inds_te[:3,-3:])
    # args = th.where(th.abs(score_gt - score_te)>0.1)
    # print(score_gt[args])
    # print(score_te[args])
    # print(args)
    # print(th.unique(args[0]))

    # -- compare --
    args0 = th.where(th.logical_not(th.isinf(score_gt))) # remove all inf
    args1 = th.where(th.logical_not(th.isinf(score_te))) # remove all inf
    diff = th.abs(score_te - score_gt) / (score_gt.abs() + 1e-5)
    diff = diff[args0]
    # print(th.abs(args0[0] - args1[0]).sum())
    # print(th.abs(args0[1] - args1[1]).sum())

    tol = 1e-5
    error = diff.mean().item()
    if error > tol: print("error: ",error)
    assert error < tol

    tol = 1e-4
    max_error = diff.max().item()
    if max_error > tol: print("max error: ",max_error)
    assert max_error < tol
