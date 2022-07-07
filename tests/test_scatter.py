
# -- python --
import sys,pytest
import numba as nb

# -- data mgnmt --
from pathlib import Path
from easydict import EasyDict as edict

# -- testing --
import unittest

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- dnls --
import dnls

# -- testing --
import torch.nn.functional as nnf
from torch.nn.functional import fold,unfold,pad
from torchvision.transforms.functional import center_crop


# -- paths --
SAVE_DIR = Path("./output/tests/")

def pytest_generate_tests(metafunc):
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)
    test_lists = {"exact":[True,False],"ps":[7],"pt":[1],
                  "ws":[10],"wt":[5],"k":[10]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

#
# -- Test Simple Scatter --
#

# @pytest.mark.skip()
def test_simple_scatter():

    # -- get args --
    dname,sigma,comp_flow,args = setup()

    # -- init vars --
    device = args.device
    clean_flow = True
    comp_flow = False

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext="jpg")
    vid = th.from_numpy(vid).to(device)
    noisy = vid + sigma * th.randn_like(vid)
    flow = dnls.testing.flow.get_flow(comp_flow,clean_flow,noisy,vid,sigma)

    # -- unpack params --
    k,ps,pt = args.k,args.ps,args.pt
    ws,wt,chnls = args.ws,args.wt,1

    # -- batching info --
    device = noisy.device
    shape = noisy.shape
    t,c,h,w = shape
    npix = t * h * w
    qStride,qSize = 1,100
    nsearch = (npix-1) // qStride + 1
    nbatches = (nsearch-1) // qSize + 1
    vid = vid.contiguous()
    th.cuda.synchronize()

    # -- nbatches --
    for index in range(nbatches):

        # -- get [patches & nlInds] --
        queryInds = dnls.utils.inds.get_query_batch(index,qSize,qStride,
                                                    t,h,w,device)
        nlDists,nlInds = dnls.simple.search.run(vid,queryInds,
                                                flow,k,ps,pt,ws,wt,chnls)

        # -- exec scatter fxns --
        # scatter_nl = dnls.scatter.ScatterNl(ps,pt,btype="simple",device=device)

        # -- testing forward --
        # patches_nl_fwd = scatter_nl(vid,nlInds)
        patches_nl_fwd = dnls.simple.scatter.run(vid,nlInds,ps)
        patches_simp_fwd = dnls.simple.scatter.run(vid,nlInds,ps)
        error = th.mean((patches_nl_fwd - patches_simp_fwd)**2).item()
        assert error < 1e-10
    th.cuda.synchronize()
    nb.cuda.synchronize()

#
# -- Test Simple Scatter --
#

# @pytest.mark.skip()
def test_efficient_scatter():

    # -- get args --
    dname,sigma,comp_flow,args = setup()

    # -- init vars --
    device = args.device
    clean_flow = True
    comp_flow = False

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext="jpg")
    vid = th.from_numpy(vid).to(device)
    noisy = vid + sigma * th.randn_like(vid)
    flow = dnls.testing.flow.get_flow(comp_flow,clean_flow,noisy,vid,sigma)

    # -- unpack params --
    k,ps,pt = args.k,args.ps,args.pt
    ws,wt,chnls = args.ws,args.wt,1

    # -- batching info --
    device = noisy.device
    shape = noisy.shape
    t,c,h,w = shape
    npix = t * h * w
    qStride,qSize = 1,100
    nsearch = (npix-1) // qStride + 1
    nbatches = (nsearch-1) // qSize + 1
    vid = vid.contiguous()
    th.cuda.synchronize()

    # -- nbatches --
    for index in range(nbatches):

        # -- get [patches & nlInds] --
        queryInds = dnls.utils.inds.get_query_batch(index,qSize,qStride,
                                                    t,h,w,device)
        nlDists,nlInds = dnls.simple.search.run(vid,queryInds,
                                                flow,k,ps,pt,ws,wt,chnls)

        # -- exec scatter fxns --
        scatter_nl = dnls.scatter.ScatterNl(ps,pt,btype="eff")

        # -- testing forward --
        patches_nl_fwd = scatter_nl(vid,nlInds)
        patches_simp_fwd = dnls.simple.scatter.run(vid,nlInds,ps)
        error = th.mean((patches_nl_fwd - patches_simp_fwd)**2).item()
        assert error < 1e-10
    th.cuda.synchronize()
    nb.cuda.synchronize()

#
# -- Test Scatter & Unfold --
#

# @pytest.mark.skip()
def test_nn_scatter(exact,ps,pt,k):

    # -- get args --
    dname,sigma,comp_flow,args = setup()

    # -- init vars --
    device = args.device
    clean_flow = True
    comp_flow = False
    dil = 1

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext="jpg")
    vid = th.from_numpy(vid).to(device)
    noisy = vid + sigma * th.randn_like(vid)
    flow = dnls.testing.flow.get_flow(comp_flow,clean_flow,noisy,vid,sigma)
    noisy = th.randn((3,64,128,128)).to(vid.device)
    vid = noisy.clone()
    th.cuda.synchronize()

    # -- unpack params --
    ws,wt,chnls = args.ws,args.wt,1

    # -- batching info --
    device = noisy.device
    shape = noisy.shape
    t,c,h,w = shape
    npix = t * h * w
    qStride,qSize = 1,npix
    nsearch = (npix-1) // qStride + 1
    nbatches = (nsearch-1) // qSize + 1
    vid = vid.contiguous()

    # -- exec scatter fxns --
    scatter_nl = dnls.scatter.ScatterNl(ps,pt,exact=exact)

    # -- get [patches & nlInds] --
    index = 0
    queryInds = dnls.utils.inds.get_query_batch(index,qSize,qStride,
                                                t,h,w,device)
    nlDists,nlInds = dnls.simple.search.run(vid,queryInds,
                                            flow,k,ps,pt,ws,wt,chnls)


    #
    # -- test logic --
    #

    # -- prepare videos --
    vid_nn = vid
    vid_nl = vid.clone()
    vid_nn.requires_grad_(True)
    vid_nl.requires_grad_(True)

    # -- run forward --
    patches_nn = run_unfold(vid_nn,ps,dil=dil)
    patches_nl = scatter_nl(vid_nl,nlInds[:,[0]])
    # print(vid_nn.shape)
    # print(nlInds.shape)
    # print(patches_nn.shape)
    # print(patches_nl.shape)
    # print(nlInds[:3])

    # -- run backward --
    patches_grad = th.randn_like(patches_nn)
    th.autograd.backward(patches_nn,patches_grad)
    th.autograd.backward(patches_nl,patches_grad)

    # -- check forward --
    diff = th.abs(patches_nn - patches_nl)
    diff = rearrange(diff,'nq k t c h w -> nq (k t c h w)')
    error = th.mean((patches_nn - patches_nl)**2).item()
    assert error < 1e-10

    # -- check backward --
    grad_nn = vid_nn.grad
    grad_nl = vid_nl.grad
    # print(patches_grad[0,0,0,0])
    # print(patches_nn[0,0,0,0])
    # print(patches_nl[0,0,0,0])
    # print(grad_nn[0,0,:3,:3])
    # print(grad_nl[0,0,:3,:3])
    # print(grad_nn[0,0,-3:,-3:])
    # print(grad_nl[0,0,-3:,-3:])

    # -- vis --
    diff = th.abs(grad_nn - grad_nl)
    if diff.max() > 1e-8: diff /= diff.max()
    print("diff.shape: ",diff.shape)
    dnls.testing.data.save_burst(diff[:,:3],'./output/tests/','diff')

    # -- test --
    if exact: tol = 1e-6
    else: tol = 1.
    error = th.mean(th.abs(grad_nn - grad_nl)/(grad_nn.abs() + 1e-8)).item()
    print("Mean Error: ",error)
    assert error < tol

    # -- viz --
    args = th.where(th.abs(grad_nn)>1.)
    error = th.abs(grad_nn[args] - grad_nl[args])/grad_nn[args].abs()
    args2 = th.where(error > 5.)
    # print(args)
    # print(th.stack([grad_nn[args][args2],grad_nl[args][args2]],-1))

    # -- error --
    if exact: tol = 1e-1
    else: tol = 1.
    args = th.where(th.abs(grad_nn)>1.)
    error = th.max(th.abs(grad_nn[args] - grad_nl[args])/grad_nn[args].abs()).item()
    print("Max Error: ",error)
    assert error < tol
    th.cuda.synchronize()
    nb.cuda.synchronize()

#
# -- Misc --
#

def setup():

    # -- set device --
    device = "cuda:0"
    th.cuda.set_device(device)

    # -- set seed --
    seed = 123
    th.cuda.set_device(device)
    th.manual_seed(seed)
    np.random.seed(seed)

    # -- options --
    comp_flow = False

    # -- init save path --
    save_dir = SAVE_DIR
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    th.cuda.synchronize()

    # -- exec test 1 --
    sigma = 50.
    dname = "text_tourbus_64"
    dname = "davis_baseball_64x64"
    args = edict({'ps':7,'pt':1,'k':10,'ws':10,'wt':5})
    args.device = device
    return dname,sigma,comp_flow,args

def run_unfold(vid,ps,dil=1):
    lpad,rpad = ps//2,(ps-1)//2
    shape_str = 't (c h w) np -> (t np) 1 1 c h w'
    vid_pad = pad(vid,[lpad,rpad,lpad,rpad,],mode="reflect")
    patches = unfold(vid_pad,(ps,ps))
    patches = rearrange(patches,shape_str,h=ps,w=ps)
    return patches
