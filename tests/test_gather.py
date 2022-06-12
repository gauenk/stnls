
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

class TestGather(unittest.TestCase):

    #
    # -- Test Simple Scatter --
    #

    def test_simple_gather(self):

        # -- get args --
        dname,sigma,comp_flow,args = self.setup()

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

        # -- gather/scatter decl --
        scatter_nl = dnls.scatter.ScatterNl(ps,pt,exact=True,device=device)
        gather_nl = dnls.gather.GatherNl(vid.shape,device=device)

        # -- batching info --
        device = noisy.device
        shape = noisy.shape
        t,c,h,w = shape
        npix = t * h * w
        qStride,qSize = 1,64
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
            patches = scatter_nl(vid,nlInds)

            # -- reset gather --
            gather_nl.vid[...] = 0
            gather_nl.wvid[...] = 0

            # -- testing forward --
            vid_nl,wvid_nl = gather_nl(patches,nlDists,nlInds)
            vid_simp,wvid_simp = dnls.simple.gather.run(patches,nlDists,
                                                        nlInds,shape=shape)
            error = th.mean((vid_nl - vid_simp)**2).item()
            assert error < 1e-10

            # -- save --
            # vid_nl /= vid_nl.max()
            # vid_simp /= vid_simp.max()
            # dnls.testing.data.save_burst(vid_nl[[0]],SAVE_DIR,"nl_%d" % index)
            # dnls.testing.data.save_burst(vid_simp[[0]],SAVE_DIR,"simp_%d" % index)

        th.cuda.synchronize()

    #
    # -- Test Gather & Unfold --
    #

    def test_nn_gather(self):

        # -- get args --
        dname,sigma,comp_flow,args = self.setup()

        # -- init vars --
        device = args.device
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

        # -- batching info --
        device = noisy.device
        shape = noisy.shape
        t,c,h,w = shape
        npix = t * h * w
        qStride,qSize = 1,npix
        nsearch = (npix-1) // qStride + 1
        nbatches = (nsearch-1) // qSize + 1
        vid = vid.contiguous()

        # -- exec gather fxns --
        scatter_nl = dnls.scatter.ScatterNl(ps,pt,exact=True,device=device)
        gather_nl = dnls.gather.GatherNl((t,c,h,w),exact=exact,device=device)

        # -- get [patches & nlInds] --
        index = 0
        queryInds = dnls.utils.inds.get_query_batch(index,qSize,qStride,
                                                    t,h,w,device)
        nlDists,nlInds = dnls.simple.search.run(vid,queryInds,
                                                flow,k,ps,pt,ws,wt,chnls)
        patches = scatter_nl(vid,nlInds)

        #
        # -- test logic --
        #

        # -- prepare videos --
        patches_nn = patches
        patches_nl = patches.clone()
        patches_nn.requires_grad_(True)
        patches_nl.requires_grad_(True)

        # -- run forward --
        vid_nn_raw,wvid_nn = self.run_fold(patches_nn,t,h,w)
        vid_nl_raw,wvid_nl = gather_nl(patches_nl,nlDists,nlInds)

        # -- reweight --
        args = th.where(wvid_nl>0)
        vid_nl = vid_nl_raw  / wvid_nl
        vid_nn = vid_nn_raw  / wvid_nn

        # -- run backward --
        vid_grad = th.randn_like(vid)
        th.autograd.backward(vid_nn,vid_grad)
        th.autograd.backward(vid_nl,vid_grad)

        # -- check forward --
        diff = th.abs(vid_nn - vid_nl)
        error = th.mean((vid_nn - vid_nl)**2).item()
        assert error < 1e-10

        # -- check backward --
        grad_nn = patches_nn.grad
        grad_nl = patches_nl.grad
        if exact: tol = 1e-10
        else: tol = 1.
        error = th.mean((grad_nn - grad_nl)**2).item()
        assert error < tol

    #
    # -- Launcher --
    #

    def setup(self):

        # -- set device --
        device = "cuda:1"
        th.cuda.set_device(device)

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
        args = edict({'ps':7,'pt':1,'k':1,'ws':10,'wt':5})
        args.device = device
        return dname,sigma,comp_flow,args

    def run_fold(self,patches,t,h,w):
        ps = patches.shape[-1]
        psHalf = ps//2
        hp,wp = h+2*psHalf,w+2*psHalf
        shape_str = '(t np) 1 1 c h w -> t (c h w) np'
        patches = rearrange(patches,shape_str,t=t)
        ones = th.ones_like(patches)

        vid_pad = fold(patches,(hp,wp),(ps,ps))
        vid = center_crop(vid_pad,(h,w))
        wvid_pad = fold(ones,(hp,wp),(ps,ps))
        wvid = center_crop(wvid_pad,(h,w))

        return vid,wvid

    def run_unfold(self,vid,ps):
        psHalf = ps//2
        shape_str = 't (c h w) np -> (t np) 1 1 c h w'
        vid_pad = pad(vid,4*[psHalf,],mode="reflect")
        patches = unfold(vid_pad,(ps,ps))
        patches = rearrange(patches,shape_str,h=ps,w=ps)
        return patches

