
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
    # test_lists = {"ps":[8],"stride":[8],"dilation":[1],
    #               "top":[0],"btm":[64],"left":[0],"right":[64]}
    test_lists = {"ps":[3,4,5,6,7,8],"stride":[1,2,3,4,5,8],"dilation":[1,2,3,4,5,8],
                  "top":[1,11],"btm":[50,57],"left":[3,7],"right":[57,30]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)
#
# -- Test Against Pytorch.nn.fold --
#

def test_nn_with_unfold(ps,stride,dilation):

    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    chnls,k,pt = 1,1,1
    ws,wt = 10,0
    adj = True
    top,btm,left,right = 0,64,0,64 # full image

    # -- sub square --
    coords = [top,left,btm,right]
    sq_h = coords[2] - coords[0]
    sq_w = coords[3] - coords[1]

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    exact = True

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device).contiguous()
    vid = th.ones_like(vid)

    # -- compute optical flow --
    flow = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- image params --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    nframes,height,width = t,h,w
    vshape = vid.shape

    # -- num of steps each direction --
    npix = t * h * w
    n_h = (sq_h - (ps-1)*dil - 1)//stride + 1
    n_w = (sq_w - (ps-1)*dil - 1)//stride + 1

    # -- skip if invalid shape --
    # valid_h = (sq_h - (ps-1)*dil - 1) % stride == 0
    # valid_w = (sq_w - (ps-1)*dil - 1) % stride == 0
    # valid = valid_h and valid_w
    # if not(valid):
    #     print("invalid: ",ps,dil,stride,coords)


    #
    # -- test logic --
    #

    # -- run unfold --
    patches_nl = run_unfold(vid,ps,stride,dil)
    patches_nn = patches_nl.clone()
    patches_nl.requires_grad_(True)
    patches_nn.requires_grad_(True)

    # -- run forward --
    vid_nn,_ = run_fold(patches_nn,t,sq_h,sq_w,stride,dil,adj)
    fold_nl = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dil,adj=adj)
    vid_nl = fold_nl(patches_nl,0)#[:,:,top:btm,left:right]

    vid_nn_s  = vid_nn /vid_nn.max()
    vid_nl_s = vid_nl / vid_nl.max()
    # dnls.testing.data.save_burst(vid_nn_s,"./output/","vid_nn")
    # dnls.testing.data.save_burst(vid_nl_s,"./output/","vid_nl")

    # -- run backward --
    vid_grad = th.randn_like(vid_nl)
    th.autograd.backward(vid_nn,vid_grad)
    th.autograd.backward(vid_nl,vid_grad)

    # -- check forward --
    delta = vid_nn - vid_nl
    error = th.sum(delta**2).item()
    assert error < 1e-10

    # -- check backward --
    grad_nn = patches_nn.grad
    grad_nl = patches_nl.grad

    # -- rearrange --
    shape_str = '(t h w) 1 1 c ph pw -> t c h w ph pw'
    grad_nn = rearrange(grad_nn,shape_str,t=t,h=n_h)
    grad_nl = rearrange(grad_nl,shape_str,t=t,h=n_h)

    # -- viz --
    diff = th.mean((grad_nn - grad_nl)**2,(-2,-1))
    diff /= diff.max()
    dnls.testing.data.save_burst(diff,"./output/","grad")

    # -- check backward --
    error = th.sum((grad_nn - grad_nl)**2).item()
    assert error < 1e-10

    # -- clean-up --
    th.cuda.empty_cache()
    del vid,flow
    del vid_nn,vid_nl
    del patches_nl,patches_nn
    del grad_nn,grad_nl,vid_grad
    th.cuda.empty_cache()
    th.cuda.synchronize()

# @pytest.mark.skip(reason="too long right now")
def test_nn(ps,stride,dilation,top,btm,left,right):

    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    chnls,k,pt = 1,1,1
    ws,wt = 10,0
    adj = False

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    exact = True

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device).contiguous()

    # -- compute optical flow --
    flow = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- image params --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    nframes,height,width = t,h,w
    vshape = vid.shape

    # -- sub square --
    coords = [top,left,btm,right]
    sq_h = coords[2] - coords[0]
    sq_w = coords[3] - coords[1]

    # -- batching info --
    npix = t * h * w
    nh = (sq_h-1)//stride+1
    nw = (sq_w-1)//stride+1
    qTotal = t * nh * nw
    qSize = qTotal
    nbatches = (qTotal-1) // qSize + 1

    # -- exec fold fxns --
    scatter_nl = dnls.scatter.ScatterNl(ps,pt,dilation=dil,exact=True)
    fold_nl = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dil,adj=adj)

    # -- patches for ifold --
    index = 0
    queryInds = dnls.utils.inds.get_iquery_batch(index,qSize,stride,
                                                 coords,t,device)
    nlDists,nlInds = dnls.simple.search.run(vid,queryInds,flow,k,
                                            ps,pt,ws,wt,chnls,
                                            stride=stride,dilation=dil)
    assert th.sum(queryInds - nlInds[:,0]) < 1e-10
    patches_nl = scatter_nl(vid,nlInds)
    patches_nn = patches_nl.clone()
    patches_nn = patches_nn.requires_grad_(True)
    patches_nl = patches_nl.requires_grad_(True)

    #
    # -- test logic --
    #

    # -- run forward --
    top,left,btm,right = coords
    vid_nn,_ = run_fold(patches_nn,t,sq_h,sq_w,stride,dil,adj)
    vid_nl = fold_nl(patches_nl,0)[:,:,top:btm,left:right]

    # -- run backward --
    vid_grad = th.randn_like(vid_nl)
    th.autograd.backward(vid_nn,vid_grad)
    th.autograd.backward(vid_nl,vid_grad)

    # -- check forward --
    delta = vid_nn - vid_nl
    error = th.sum(delta**2).item()
    assert error < 1e-10

    # -- check backward --
    grad_nn = patches_nn.grad
    grad_nl = patches_nl.grad

    # -- rearrange --
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
    del queryInds,nlDists,nlInds
    th.cuda.empty_cache()
    th.cuda.synchronize()

#
# -- Test a Batched Ours Against Pytorch.nn.fold --
#

# @pytest.mark.skip(reason="too long right now")
def test_batched(ps,stride,dilation,top,btm,left,right):

    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    chnls,k,pt = 1,1,1
    ws,wt = 10,0

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
    flow = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)
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
    qTotal = t * nh * nw
    qSize = 512
    nbatches = (qTotal-1) // qSize + 1

    # -- exec fold fxns --
    scatter_nl = dnls.scatter.ScatterNl(ps,pt,dilation=dil,exact=True)
    fold_nl = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dil,adj=adj)
    patches_nl = []
    gpu_mem.print_gpu_stats(gpu_stats,"pre-loop")

    for index in range(nbatches):

        # -- batch info --
        qindex = min(qSize * index,npix)
        qSize =  min(qSize, qTotal - qindex)

        # -- get patches --
        queryInds = dnls.utils.inds.get_iquery_batch(qindex,qSize,stride,
                                                     coords,t,device)
        nlDists,nlInds = dnls.simple.search.run(vid,queryInds,flow,k,
                                                ps,pt,ws,wt,chnls,
                                                stride=stride,dilation=dil)
        patches_nl_i = scatter_nl(vid,nlInds)
        del queryInds,nlDists,nlInds
        th.cuda.empty_cache()

        patches_nl_i = patches_nl_i.requires_grad_(True)

        # -- run forward --
        vid_nl = fold_nl(patches_nl_i,qindex)
        patches_nl.append(patches_nl_i)

    # -- vis --
    gpu_mem.print_gpu_stats(gpu_stats,"post-loop")

    # -- forward all at once --
    index,qSize = 0,qTotal
    queryInds = dnls.utils.inds.get_iquery_batch(index,qSize,stride,
                                                 coords,t,device)
    nlDists,nlInds = dnls.simple.search.run(vid,queryInds,flow,k,
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
    del queryInds,nlDists,nlInds
    th.cuda.empty_cache()
    th.cuda.synchronize()

# @pytest.mark.skip(reason="too long right now")
def test_shifted(ps,stride,dilation,top,btm,left,right):

    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    chnls,k,pt = 1,1,1
    ws,wt = 10,0
    shift = 2

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    exact = True

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device).contiguous()
    flow = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- image params --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    nframes,height,width = t,h,w
    vshape = vid.shape

    # -- sub square --
    coords = [top,left,btm,right]
    sq_h = coords[2] - coords[0]
    sq_w = coords[3] - coords[1]

    # -- shifted sub square --
    shift_coords = [x+shift for x in coords]
    # shift_vshape = (nframes,color,height+shift,width+shift)
    shift_vshape = (nframes,color,height+2*shift,width+2*shift)

    # -- batching info --
    npix = t * h * w
    n_h = (sq_h-1)//stride+1
    n_w = (sq_w-1)//stride+1
    qTotal = t * n_h * n_w
    qSize = qTotal
    nbatches = (qTotal-1) // qSize + 1

    # -- exec fold fxns --
    scatter_nl = dnls.scatter.ScatterNl(ps,pt,dilation=dil,exact=True)
    fold_nl = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dil)
    shift_fold_nl = dnls.ifold.iFold(shift_vshape,shift_coords,
                                     stride=stride,dilation=dil)

    # -- patches for ifold --
    index = 0
    queryInds = dnls.utils.inds.get_iquery_batch(index,qSize,stride,
                                                 coords,t,device)
    nlDists,nlInds = dnls.simple.search.run(vid,queryInds,flow,k,
                                            ps,pt,ws,wt,chnls,
                                            stride=stride,dilation=dil)
    assert th.sum(queryInds - nlInds[:,0]) < 1e-10
    patches_nl = scatter_nl(vid,nlInds)
    patches_nn = patches_nl.clone()
    patches_nn = patches_nn.requires_grad_(True)
    patches_nl = patches_nl.requires_grad_(True)

    #
    # -- test logic --
    #

    # -- run forward --
    top,left,btm,right = coords
    vid_shift = shift_fold_nl(patches_nn,0)
    dnls.testing.data.save_burst(vid_shift,"./output/tests/","vid_shift")
    vid_shift = vid_shift[:,:,top+shift:btm+shift]
    vid_shift = vid_shift[:,:,:,left+shift:right+shift]
    vid_nl = fold_nl(patches_nl,0)
    dnls.testing.data.save_burst(vid_nl,"./output/tests/","vid_nl")
    vid_nl = vid_nl[:,:,top:btm,left:right]

    # -- run backward --
    vid_grad = th.randn_like(vid_nl)
    th.autograd.backward(vid_shift,vid_grad)
    th.autograd.backward(vid_nl,vid_grad)

    # -- check forward --
    delta = vid_shift - vid_nl
    error = th.sum(delta**2).item()
    assert error < 1e-10

    # -- check backward --
    grad_nn = patches_nn.grad
    grad_nl = patches_nl.grad

    # -- rearrange --
    shape_str = '(t h w) 1 1 c ph pw -> t c h w ph pw'
    grad_nn = rearrange(grad_nn,shape_str,t=t,h=n_h)
    grad_nl = rearrange(grad_nl,shape_str,t=t,h=n_h)

    # -- check backward --
    error = th.sum((grad_nn - grad_nl)**2).item()
    assert error < 1e-10

    # -- clean-up --
    th.cuda.empty_cache()
    del vid,flow
    del vid_shift,vid_nl
    del patches_nl,patches_nn
    del grad_nn,grad_nl,vid_grad
    del queryInds,nlDists,nlInds
    th.cuda.empty_cache()
    th.cuda.synchronize()

def test_shrink_search():

    # -- get args --
    ps,stride,dilation = 5,2,1
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    chnls,k,pt = 1,1,1
    ws,wt = 10,0

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    exact = True
    exact = True
    gpu_stats = False

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device).contiguous()
    flow = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape

    # -- sub square --
    top,left,btm,right = 0,0,h,w
    coords = [top,left,btm,right]
    sq_h = coords[2] - coords[0]
    sq_w = coords[3] - coords[1]

    # -- batching info --
    npix = t * h * w
    n_h = (sq_h-1)//stride+1
    n_w = (sq_w-1)//stride+1
    qTotal = t * n_h * n_w
    qSize = qTotal
    nbatches = (qTotal-1) // qSize + 1

    # -- padded video --
    # padf = 14 # something big
    # vid_pad = pad(vid,[padf,]*4,mode="reflect")

    # -- get folds --
    scatter_nl = dnls.scatter.ScatterNl(ps,pt,dilation=dil,exact=True)
    fold_nl = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dil)
    wfold_nl = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dil)

    # -- get patches with dilation --
    qindex,k,pt,chnls = 0,1,1,1
    queryInds = dnls.utils.inds.get_iquery_batch(qindex,qSize,stride,
                                                 coords,t,device)
    nlDists,nlInds = dnls.simple.search.run(vid,queryInds,flow,
                                            k,ps,pt,ws,wt,chnls,
                                            stride=stride,dilation=dil)
    patches = scatter_nl(vid,nlInds[:,[0]])
    ones = th.ones_like(patches)
    vid_f = fold_nl(patches,0)
    wvid_f = wfold_nl(ones,0)
    vid_f /= wvid_f

    # -- inspect --
    # dnls.testing.data.save_burst(vid_f,"./output/tests/","vid_f")
    # diff = th.abs(vid-vid_f)
    # if diff.max() > 1e-3: diff /= diff.max()
    # dnls.testing.data.save_burst(diff,"./output/tests/","diff_f")

    # -- misc --
    error = th.sum((vid_f - vid)**2).item()
    assert error < 1e-10
    th.cuda.synchronize()

def run_fold(_patches,_t,_h,_w,_stride=1,_dil=1,_adj=False):
    # -- avoid pytest fixtures --
    patches = _patches
    t,h,w = _t,_h,_w
    stride,dil,adj = _stride,_dil,_adj

    # -- unpack --
    ps = patches.shape[-1]
    padf_lg,padf_sm = dil * (ps//2),dil * ((ps-1)//2)
    if adj is True: padf_lg,padf_sm = 0,0
    hp,wp = h+padf_lg+padf_sm,w+padf_lg+padf_sm
    shape_str = '(t np) 1 1 c h w -> t (c h w) np'
    patches = rearrange(patches,shape_str,t=t)
    ones = th.ones_like(patches)

    # -- folded --
    vid_pad = fold(patches,(hp,wp),(ps,ps),stride=stride,dilation=dil)
    vid = vid_pad[:,:,padf_lg:h+padf_lg,padf_lg:w+padf_lg]

    # -- weigthed vid --
    wvid_pad = fold(ones,(hp,wp),(ps,ps),stride=stride,dilation=dil)
    wvid = wvid_pad[:,:,padf_lg:h+padf_lg,padf_lg:w+padf_lg]

    return vid,wvid

def run_unfold(_vid,_ps,_stride=1,_dil=1):

    # -- avoid fixutres --
    vid,stride = _vid,_stride
    ps,dil = _ps,_dil

    # -- padding --
    # padf_lg,padf_sm = dil * (ps//2),dil * ((ps-1)//2)
    # # if adj is True: padf_lg,padf_sm = 0,0
    # psHalf = ps//2
    # padf = dil * psHalf
    # vid_pad = pad(vid,4*[padf,],mode="reflect")

    # -- unfold --
    shape_str = 't (c h w) np -> (t np) 1 1 c h w'
    patches = unfold(vid,(ps,ps),stride=stride,dilation=dil)
    patches = rearrange(patches,shape_str,h=ps,w=ps)

    return patches

