
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
from dnls.utils.pads import comp_pads

# -- meshgrid --
import cache_io

# -- test func --
from torch.nn.functional import fold,unfold,pad
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/")

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
                  "ws":[-1],"top":[0],"btm":[64],"left":[0],"right":[64]}
    # test_lists = {"ps":[3,4,5,6,7,8],"stride":[1,2,3,4,5,8],"dilation":[1,2,3,4,5,8],
    #               "top":[1,11],"btm":[50,57],"left":[3,7],"right":[57,30]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

#
#
# -- Primary Testing Class --
#
#

@pytest.mark.skip(reason="too long right now")
def test_nn(ps,stride,dilation,top,btm,left,right):


    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    chnls,k,pt = 1,1,1
    ws,wt = 10,0
    ws = -1
    k = -1

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    exact = True
    exact = True
    gpu_stats = False
    adj = False

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device)[:1,].contiguous()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-2)
    vid = th.cat([vid,vid],-2)

    # -- compute flow --
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape

    # -- sub square --
    coords = [top,left,btm,right]
    sq_h = coords[2] - coords[0]
    sq_w = coords[3] - coords[1]

    # -- batching info --
    npix = t * h * w
    nh = (sq_h-1)//stride+1
    nw = (sq_w-1)//stride+1
    ntotal = t * nh * nw
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- exec fold fxns --
    xsearch = dnls.xsearch.CrossSearchNl(flows.fflow, flows.bflow, k, ps, pt,
                                           ws, wt, chnls=chnls,dilation=dil, stride=stride)
    scatter_nl = dnls.scatter.ScatterNl(ps,pt,dilation=dil,exact=True)
    fold_nl = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dil,adj=adj)
    patches_nl = []
    gpu_mem.print_gpu_stats(gpu_stats,"pre-loop")

    # -- query inds
    qindex = 0
    iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch,stride,
                                                 coords,t,device)
    # -- run search --
    nlDists_nn,nlInds_nn = dnls.simple.xsearch_nn.run_nn(vid,ps,stride=stride,dilation=dil)
    nlDists_simp,nlInds_simp = dnls.simple.xsearch_nn.run(vid,ps,stride=stride,dilation=dil)

    # -- compare --
    error = th.sum(th.abs(nlDists_simp - nlDists_nn)).item()
    print("error: ",error)
    assert error < 1e-10

    # perc_neq = th.mean((nlInds_simp != nlInds_simp)*1.)
    # print("perc_neq: ",perc_neq)
    # assert perc_neq < 0.05

@pytest.mark.skip(reason="too long right now")
def test_simp(ps,stride,dilation,top,btm,left,right):


    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    chnls,k,pt = 3,1,1
    ws,wt = 10,0
    ws = -1
    k = -1

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    exact = True
    exact = True
    gpu_stats = False
    adj = False

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device)[:1,].contiguous()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- compute flow --
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape

    # -- sub square --
    top,left,btm,right=0,0,h,w
    coords = [top,left,btm,right]
    sq_h = coords[2] - coords[0]
    sq_w = coords[3] - coords[1]

    # -- batching info --
    stride0 = 1
    npix = t * h * w
    nh = (sq_h-1)//stride0+1
    nw = (sq_w-1)//stride0+1
    ntotal = t * nh * nw
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- exec fold fxns --
    xsearch_nl = dnls.xsearch.CrossSearchNl(flows.fflow, flows.bflow, k, ps, pt,
                                            ws, wt, chnls=chnls,dilation=dil, stride=stride)
    scatter_nl = dnls.scatter.ScatterNl(ps,pt,dilation=dil,exact=True)
    fold_nl = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dil,adj=adj)
    patches_nl = []
    gpu_mem.print_gpu_stats(gpu_stats,"pre-loop")

    # -- query inds
    qindex = 0
    iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch,stride0,
                                                coords,t,device)

    # -- run search --
    # nlDists_cu,nlInds_cu = xsearch_nl(vid,iqueries)
    nlDists_simp,nlInds_simp = dnls.simple.xsearch_nn.run(vid,ps,stride=stride,dilation=dil)
    nlDists_cu,nlInds_cu = dnls.simple.xsearch.run(vid,iqueries,flows,k,
                                                   ps,pt,ws,wt,chnls,
                                                   stride=stride,dilation=dil)
    # -- reshape --
    nlDists_cu = rearrange(nlDists_cu,'(h w) nh nw -> h w nh nw',h=h)

    # -- viz --
    # print(nlDists_simp.shape)
    # print(nlDists_simp[8,8,:3,:3])
    # print(nlDists_cu[8,8,:3,:3])
    # print(nlDists_simp[8,8,8,8])
    # print(nlDists_cu[8,8,8,8])
    # print(nlDists_cu[8,8,28:34,28:34])

    # -- compare --
    error = th.sum(th.abs(nlDists_cu - nlDists_simp)).item()
    print("error: ",error)
    assert error < 1e-10

    # perc_neq = th.mean((nlInds_cu != nlInds_simp)*1.)
    # print("perc_neq: ",perc_neq)
    # assert perc_neq < 0.05

def test_cu(ps,stride,dilation,top,btm,left,right):


    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    chnls,k,pt = 3,1,1
    ws,wt = 10,0
    ws = -1
    k = -1

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    exact = True
    exact = True
    gpu_stats = False
    adj = False

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device)[:1,].contiguous()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- compute flow --
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape

    # -- sub square --
    top,left,btm,right=0,0,h,w
    coords = [top,left,btm,right]
    sq_h = coords[2] - coords[0]
    sq_w = coords[3] - coords[1]

    # -- batching info --
    stride0 = 1
    npix = t * h * w
    nh = (sq_h-1)//stride0+1
    nw = (sq_w-1)//stride0+1
    ntotal = t * nh * nw
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- pads --
    oh0,ow0,_,_ = comp_pads(vid.shape, ps, stride0, dil)
    oh1,ow1,hp,wp = comp_pads(vid.shape, ps, stride, dil)
    n_h = (hp - (ps-1)*dil - 1)//stride + 1
    n_w = (wp - (ps-1)*dil - 1)//stride + 1

    # -- exec fold fxns --
    xsearch_nl = dnls.xsearch.CrossSearchNl(flows.fflow, flows.bflow, k, ps, pt,
                                            ws, wt, oh0, ow0, oh1, ow1,
                                            chnls=chnls,dilation=dil, stride=stride)
    scatter_nl = dnls.scatter.ScatterNl(ps,pt,dilation=dil,exact=True)
    fold_nl = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dil,adj=adj)
    patches_nl = []
    gpu_mem.print_gpu_stats(gpu_stats,"pre-loop")

    # -- query inds
    qindex = 0
    iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch,stride0,
                                                coords,t,device)

    # -- run search --
    nlDists_cu,nlInds_cu = xsearch_nl(vid,iqueries)
    # nlDists_simp,nlInds_simp = dnls.simple.xsearch_nn.run(vid,ps,stride=stride,dilation=dil)
    nlDists_simp,nlInds_simp = dnls.simple.xsearch.run(vid,iqueries,flows,k,
                                                       ps,pt,ws,wt,chnls,
                                                       stride=stride,dilation=dil)
    # nlDists_cu,nlInds_cu = dnls.simple.xsearch.run(vid,iqueries,flows,k,
    #                                                ps,pt,ws,wt,chnls,
    #                                                stride=stride,dilation=dil)
    # -- reshape --
    if nlDists_simp.ndim == 3:
        nlDists_simp = rearrange(nlDists_simp,'(h w) nh nw -> h w nh nw',h=h)
    nlDists_cu = rearrange(nlDists_cu,'(h w) nh nw -> h w nh nw',h=h)

    # -- viz --
    print(nlDists_simp.shape)
    print(nlDists_simp[8,8,:3,:3])
    print(nlDists_cu[8,8,:3,:3])
    print(nlDists_simp[8,8,8,8])
    print(nlDists_cu[8,8,8,8])
    print(nlDists_cu[8,8,28:34,28:34])

    # -- compare --
    error = th.sum(th.abs(nlDists_cu - nlDists_simp)).item()
    print("error: ",error)
    assert error < 1e-10

    # perc_neq = th.mean((nlInds_cu != nlInds_simp)*1.)
    # print("perc_neq: ",perc_neq)
    # assert perc_neq < 0.05

@pytest.mark.skip(reason="too long right now")
def test_batched(ps,stride,dilation,top,btm,left,right,ws,wt):

    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    chnls,k,pt = 1,1,1
    ws,wt = 10,0
    ws = -1
    k = -1


    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    exact = True
    exact = True
    gpu_stats = False
    adj = False

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device).contiguous()
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape

    # -- sub square --
    coords = [top,left,btm,right]
    sq_h = coords[2] - coords[0]
    sq_w = coords[3] - coords[1]

    # -- batching info --
    npix = t * h * w
    nh = (sq_h-1)//stride+1
    nw = (sq_w-1)//stride+1
    ntotal = t * nh * nw
    nbatch = 512
    nbatches = (ntotal-1) // nbatch + 1

    # -- exec fold fxns --
    search_nl = dnls.xsearch.CrossSearchNl(flows.fflow, flows.bflow, k, ps, pt,
                                           ws, wt, chnls=chnls,dilation=dil, stride=stride)
    scatter_nl = dnls.scatter.ScatterNl(ps,pt,dilation=dil,exact=True)
    fold_nl = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dil,adj=adj)
    patches_nl = []
    gpu_mem.print_gpu_stats(gpu_stats,"pre-loop")

    for index in range(nbatches):

        # -- batch info --
        qindex = min(nbatch * index,npix)
        nbatch =  min(nbatch, ntotal - qindex)

        # -- get patches --
        iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch,stride,
                                                     coords,t,device)
        nlDists,nlInds = dnls.simple.xsearch.run(vid,iqueries,flows,k,
                                                 ps,pt,ws,wt,chnls,
                                                 stride=stride,dilation=dil)
        nlDists,nlInds = search_nl(vid,iqueries,flows,k,
                                   ps,pt,ws,wt,chnls,
                                   stride=stride,dilation=dil)
        patches_nl_i = scatter_nl(vid,nlInds)
        del iqueries,nlDists,nlInds
        th.cuda.empty_cache()

        patches_nl_i = patches_nl_i.requires_grad_(True)

        # -- run forward --
        vid_nl = fold_nl(patches_nl_i,qindex)
        patches_nl.append(patches_nl_i)

    # -- vis --
    gpu_mem.print_gpu_stats(gpu_stats,"post-loop")

    # -- forward all at once --
    index,nbatch = 0,ntotal
    iqueries = dnls.utils.inds.get_iquery_batch(index,nbatch,stride,
                                                 coords,t,device)
    nlDists,nlInds = dnls.simple.xsearch.run(vid,iqueries,flow,k,
                                             ps,pt,ws,wt,chnls,
                                             stride=stride,dilation=dil)
    patches_nn = scatter_nl(vid,nlInds)
    patches_nn.requires_grad_(True)
    gpu_mem.print_gpu_stats(gpu_stats,"post-search")
    vid_nn,_ = run_fold(patches_nn,t,sq_h,sq_w,stride,dil,adj)
    gpu_mem.print_gpu_stats(gpu_stats,"post-fold")

    # -- run backward --
    top,left,btm,right = coords
    vid_grad = th.randn_like(vid_nn)
    vid_nl = fold_nl.vid[:,:,top:btm,left:right]
    th.autograd.backward(vid_nn,vid_grad)
    th.autograd.backward(vid_nl,vid_grad)
    gpu_mem.print_gpu_stats(gpu_stats,"post-bkw")
    # dnls.testing.data.save_burst(vid_nn,"./output/","vid_nn")
    # dnls.testing.data.save_burst(vid_nl,"./output/","vid_nl")

    # -- get grads --
    grad_nn = patches_nn.grad
    grad_nl = th.cat([p_nl.grad for p_nl in patches_nl])

    # -- check forward --
    top,left,btm,right = coords
    error = th.sum((vid_nn - vid_nl)**2).item()
    assert error < 1e-10

    # -- reshape --
    shape_str = '(t h w) 1 1 c ph pw -> t c h w ph pw'
    grad_nn = rearrange(grad_nn,shape_str,t=t,h=nh)
    grad_nl = rearrange(grad_nl,shape_str,t=t,h=nh)


    # -- check backward --
    error = th.sum((grad_nn - grad_nl)**2).item()
    assert error < 1e-10

    # -- clean-up --
    th.cuda.empty_cache()
    del vid,flow
    del vid_nn,vid_nl
    del patches_nl,patches_nn
    del grad_nn,grad_nl,vid_grad
    del iqueries,nlDists,nlInds
    th.cuda.empty_cache()
    th.cuda.synchronize()

