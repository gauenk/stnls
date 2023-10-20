
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

# -- stnls --
import stnls
import stnls.utils.gpu_mem as gpu_mem

# -- meshgrid --


# -- test func --
from torch.nn.functional import fold,unfold,pad
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/")

def pytest_generate_tests(metafunc):
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)
    test_lists = {"ps":[3],"stride":[1],"dilation":[1]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)
#
# -- Test Against Pytorch.nn.fold --
#

def test_nn(ps,stride,dilation):

    # -- get args --
    dil = dilation
    dnames,ext = ["davis_baseball_64x64",],"jpg"
    chnls,k,pt = 1,1,1
    ws,wt = 10,0
    adj = True

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    exact = True

    # -- load data --
    vid = stnls.testing.data.load_burst_batch("./data/",dnames,ext=ext)
    vid = vid.to(device).contiguous()
    # vid = th.ones_like(vid)

    # -- compute optical flow --
    flow = stnls.flow.get_flow_batch(comp_flow,clean_flow,vid,vid,0.)

    # -- image params --
    device = vid.device
    shape = vid.shape
    # B,t,color,h,w = shape
    B,T,F,H,W = shape
    nframes,height,width = T,H,W
    vshape = vid.shape
    nH = (H-1)//stride+1
    nW = (W-1)//stride+1

    #
    # -- test logic --
    #

    # -- run unfold --
    patches_nl = []
    for b in range(B):
        patches_nl_b = run_unfold(vid[b],ps,stride,dil)
        patches_nl.append(patches_nl_b)
    patches_nl = th.stack(patches_nl)
    patches_nn = patches_nl.clone()
    patches_nl.requires_grad_(True)
    patches_nn.requires_grad_(True)
    # print(patches_nn.shape,(nH,nW))

    # -- run forward --
    B = patches_nn.shape[0]
    vid_nn,wvid_nn = [],[]
    for b in range(B):
        vid_nn_b,wvid_nn_b = run_fold(patches_nn[b],T,H,W,stride,dil)
        vid_nn.append(vid_nn_b/wvid_nn_b)
        wvid_nn.append(wvid_nn_b)
    vid_nn = th.stack(vid_nn)
    wvid_nn = th.stack(wvid_nn)
    vid_nn_s  = vid_nn /vid_nn.max()

    # print(vid[0,0,0,:5,:5])
    # print(patches_nn[0,0,0,0,0,:,:])
    # print(patches_nn[0,1,0,0,0,:,:])
    # print(vid[0,0,0,-5:,-5:])
    # print(patches_nn.shape)
    # print(patches_nn[0,64*63-1,0,0,0,:,:])
    # print(patches_nn[0,64*64-2,0,0,0,:,:])
    # print(patches_nn[0,64*64-1,0,0,0,:,:])

    # -- run fwd ours --
    fold_nl = stnls.NlFold(vshape,stride=stride,dilation=dil)
    vid_nl = fold_nl(patches_nl[:,:,0].contiguous())#[:,:,top:btm,left:right]
    vid_nl_s = vid_nl / vid_nl.max()
    print("vid_nl_s.shape: ",vid_nl_s.shape)
    print("vid_nn_s.shape: ",vid_nn_s.shape)
    # print(vid_nl_s)
    # print(vid_nn_s)
    stnls.testing.data.save_burst(vid_nn_s[0],"./output/tests/tile/","vid_nn")
    stnls.testing.data.save_burst(vid_nl_s[0],"./output/tests/tile/","vid_nl")


    # -- run backward --
    vid_grad = th.randn_like(vid_nl)
    # vid_grad = th.ones_like(vid_nl)
    th.autograd.backward(vid_nn,vid_grad)
    th.autograd.backward(vid_nl,vid_grad)

    # -- check forward --
    delta = vid_nn - vid_nl
    error = th.sum(delta**2).item()
    assert error < 1e-10

    # -- check backward --
    grad_nn = patches_nn.grad
    grad_nl = patches_nl.grad

    # -- viz --
    # print(grad_nn.shape)
    # qi_grid = [0,1,2,3,64*32+30]
    # for qi in qi_grid:
    #     print(qi)
    #     print(grad_nn[0,qi,0,0,0])
    #     print(grad_nl[0,qi,0,0,0])

    # -- rearrange --
    # shape_str = 'b (t h w) 1 1 c ph pw -> b t c h w ph pw'
    # grad_nn = rearrange(grad_nn,shape_str,t=T,h=nH)
    # grad_nl = rearrange(grad_nl,shape_str,t=T,h=nH)

    # -- viz --
    # diff = th.mean((grad_nn - grad_nl)**2,(-2,-1))
    # diff /= diff.max()
    # # stnls.testing.data.save_burst(diff[0],"./output/","grad")

    # -- check backward --
    args_nz = th.where(grad_nn>0)
    error = th.sum((grad_nn[args_nz] - grad_nl[args_nz])**2).item()
    assert error < 1e-10

    # # -- clean-up --
    # th.cuda.empty_cache()
    # del vid,flow
    # del vid_nn,vid_nl
    # del patches_nl,patches_nn
    # del grad_nn,grad_nl,vid_grad
    # del fold_nl
    # th.cuda.empty_cache()
    # th.cuda.synchronize()

def run_fold(_patches,_t,_h,_w,_stride=1,_dil=1,_adj=False):

    # -- avoid pytest fixtures --
    patches = _patches
    t,h,w = _t,_h,_w
    stride,dil,adj = _stride,_dil,_adj

    # -- unpack --
    ps = patches.shape[-1]
    padf_lg,padf_sm = dil * (ps//2),dil * ((ps-1)//2)
    # padf_lg,padf_sm = dil * (ps//2),dil * ((ps-1)//2)
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
    padf_lg,padf_sm = dil * (ps//2),dil * ((ps-1)//2)
    # if adj is True: padf_lg,padf_sm = 0,0
    psHalf = ps//2
    padf = dil * psHalf
    vid_pad = pad(vid,4*[padf,],mode="reflect")

    # -- unfold --
    shape_str = 't (c h w) np -> (t np) 1 1 c h w'
    patches = unfold(vid_pad,(ps,ps),stride=stride,dilation=dil)
    patches = rearrange(patches,shape_str,h=ps,w=ps)

    return patches

