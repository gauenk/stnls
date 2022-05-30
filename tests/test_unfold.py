
# -- python --
import sys

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

# -- test func --
import torch.nn.functional as nnf
from torch.nn.functional import fold,unfold,pad
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/")

#
# -- Primary Testing Class --
#

class TestUnfold(unittest.TestCase):

    #
    # -- Test v.s. NN --
    #

    def test_nn_unfold(self):

        # -- get args --
        dname,sigma,comp_flow,args = self.setup()

        # -- init vars --
        device = "cuda:0"
        clean_flow = True
        comp_flow = False
        exact = True

        # -- load data --
        vid = dnls.testing.data.load_burst("./data/",dname,ext="jpg")
        vid = th.from_numpy(vid).to(device)
        noisy = vid + sigma * th.randn_like(vid)
        flow = dnls.testing.flow.get_flow(comp_flow,clean_flow,noisy,vid,sigma)

        # -- unpack params --
        k,ps,pt = args.k,args.ps,args.pt
        ws,wt,chnls = args.ws,args.wt,1
        dil = args.dilation

        # -- batching info --
        device = noisy.device
        shape = noisy.shape
        t,c,h,w = shape
        npix = t * h * w
        qStride,qSize = 1,npix
        nsearch = (npix-1) // qStride + 1
        nbatches = (nsearch-1) // qSize + 1
        vid = vid.contiguous()
        vid = th.randn_like(vid)

        # -- exec unfold fxns --
        scatter_nl = dnls.scatter.ScatterNl(ps,pt,dilation=dil,exact=True)
        unfold_nl = dnls.unfold.Unfold(ps,dilation=dil)

        #
        # -- test logic --
        #

        # -- prepare videos --
        psHalf = ps//2
        padf = psHalf*dil
        vid = pad(vid,4*[padf,],mode="reflect")
        vid_nn = vid.clone()
        vid_nl = vid.clone()
        vid_nn.requires_grad_(True)
        vid_nl.requires_grad_(True)

        # -- run forward --
        patches_nn = self.run_unfold(vid_nn,ps,dil)
        vid_nl_cc = center_crop(vid_nl,(h,w)).contiguous()
        patches_nl = unfold_nl(vid_nl_cc,0,npix)

        # -- run backward --
        patches_grad = th.ones_like(patches_nn)
        th.autograd.backward(patches_nn,patches_grad)
        th.autograd.backward(patches_nl,patches_grad)
        print("dil: ",dil)

        # -- check forward --
        error = th.sum((patches_nn - patches_nl)**2).item()
        assert error < 1e-10

        # -- check backward --
        grad_nn = center_crop(vid_nn.grad,(h,w))
        grad_nl = center_crop(vid_nl.grad,(h,w))
        if exact: tol = 1e-10
        else: tol = 1.

        diff = th.abs(grad_nn - grad_nl)
        diff /= diff.max()
        dnls.testing.data.save_burst(diff,SAVE_DIR,"diff")

        error = th.sum((grad_nn - grad_nl)**2).item()
        assert error < tol

    #
    # -- Launcher --
    #

    def setup(self):

        # -- set seed --
        seed = 123
        th.manual_seed(seed)
        np.random.seed(seed)

        # -- options --
        comp_flow = False

        # -- init save path --
        save_dir = SAVE_DIR
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # -- exec test 1 --
        sigma = 50.
        dname = "text_tourbus_64"
        dname = "davis_baseball_64x64"
        args = edict({'ps':5,'pt':1,'k':1,'ws':10,'wt':5,'dilation':1})
        return dname,sigma,comp_flow,args

    def run_fold(self,patches,t,h,w,dil=1):
        ps = patches.shape[-1]
        psHalf = ps//2
        padf = dil * psHalf
        hp,wp = h+2*padf,w+2*padf
        shape_str = '(t np) 1 1 c h w -> t (c h w) np'
        patches = rearrange(patches,shape_str,t=t)
        ones = th.ones_like(patches)

        vid_pad = fold(patches,(hp,wp),(ps,ps),dilation=dil)
        vid = center_crop(vid_pad,(h,w))
        wvid_pad = fold(ones,(hp,wp),(ps,ps),dilation=dil)
        wvid = center_crop(wvid_pad,(h,w))

        return vid,wvid

    def run_unfold(self,vid_pad,ps,dil=1):
        # psHalf = ps//2
        # vid_pad = pad(vid,4*[psHalf,],mode="reflect")
        shape_str = 't (c h w) np -> (t np) 1 1 c h w'
        patches = unfold(vid_pad,(ps,ps),dilation=dil)
        patches = rearrange(patches,shape_str,h=ps,w=ps)
        return patches

