
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
from torch.nn.functional import fold,unfold,pad
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/")

#
# -- Primary Testing Class --
#

class TestFold(unittest.TestCase):

    #
    # -- Test v.s. NN --
    #

    def test_nn_fold(self):

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

        # -- exec fold fxns --
        scatter_nl = dnls.scatter.ScatterNl(ps,pt,dilation=dil,exact=True)
        fold_nl = dnls.fold.Fold((t,c,h,w),dilation=dil)

        # -- get [patches & nlInds] --
        index = 0
        queryInds = dnls.utils.inds.get_query_batch(index,qSize,qStride,
                                                    t,h,w,device)
        nlDists,nlInds = dnls.simple.search.run(vid,queryInds,
                                                flow,k,ps,pt,ws,wt,chnls)
        patches = scatter_nl(vid,nlInds)
        # patches = self.run_unfold(vid,ps,dil)

        # -- testing mods --
        # patches_cp = patches.clone()
        # # patches_cp[...] = 1.
        # # patches_cp[1:2] = patches[1:2]
        # patches = patches_cp

        #
        # -- test logic --
        #

        # -- prepare videos --
        patches_nn = patches
        patches_nl = patches.clone()
        patches_nn.requires_grad_(True)
        patches_nl.requires_grad_(True)

        # -- run forward --
        vid_nn,_ = self.run_fold(patches_nn,t,h,w,dil)
        vid_nl = fold_nl(patches_nl,0)

        # -- run backward --
        vid_grad = th.randn_like(vid)
        th.autograd.backward(vid_nn,vid_grad)
        th.autograd.backward(vid_nl,vid_grad)

        # -- save ex --
        vid_nn_s = vid_nn / vid_nn.max()
        vid_nl_s = vid_nl / vid_nn.max()
        dnls.testing.data.save_burst(vid_nn_s,SAVE_DIR,"vid_nn")
        dnls.testing.data.save_burst(vid_nl_s,SAVE_DIR,"vid_nl")
        diff = th.abs(vid_nn_s - vid_nl_s)
        diff /= diff.max()
        dnls.testing.data.save_burst(diff,SAVE_DIR,"diff")

        # -- vis --
        print("\n")
        print(patches[0,0,0,0])
        print(patches[1,0,0,0])
        print("-"*20)
        print(vid_nn[0,0,:3,:3])
        print("-"*20)
        print(vid_nl[0,0,:3,:3])

        # -- check forward --
        error = th.sum((vid_nn - vid_nl)**2).item()
        # hm,wm = h-ps,w-ps
        # error = th.mean((center_crop(vid_nn - vid_nl,(hm,wm)))**2).item()
        assert error < 1e-10

        # -- check backward --
        grad_nn = patches_nn.grad
        grad_nl = patches_nl.grad

        # -- inspect grads --
        print("grad_nn.shape: ",grad_nn.shape)
        print(grad_nn[0,0,0,0])
        print(grad_nl[0,0,0,0])

        print(grad_nn[100,0,0,0])
        print(grad_nl[100,0,0,0])

        # -- check backward --
        if exact: tol = 1e-10
        else: tol = 1.
        error = th.mean((grad_nn - grad_nl)**2).item()
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
        args = edict({'ps':3,'pt':1,'k':1,'ws':10,'wt':5,'dilation':5})
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

    def run_unfold(self,vid,ps,dil=1):
        psHalf = ps//2
        padf = dil * psHalf
        shape_str = 't (c h w) np -> (t np) 1 1 c h w'
        vid_pad = pad(vid,4*[padf,],mode="reflect")
        patches = unfold(vid_pad,(ps,ps),dilation=dil)
        patches = rearrange(patches,shape_str,h=ps,w=ps)
        return patches

