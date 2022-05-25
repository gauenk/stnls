
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
from torch.nn.functional import fold

# # -- testing --
# from torch.nn.functional import nnf

# -- paths --
SAVE_DIR = Path("./output/tests/")

#
# -- Primary Testing Class --
#

class TestGather(unittest.TestCase):


    def get_folded(patches):
        pass

    def get_gather_func(self,gather_type,im_shape,device):
        if gather_type == "simple":
            gather_simp = dnls.simple.gather.run
            return gather_simp
        elif gather_type == "cuda":
            pass
        elif gather_type == "nl":
            gather_nl = dnls.gather.GatherNL(im_shape)
            return gather_nl
        else:
            raise ValueError(f"Uknown gather func [{gather_type}]")

    def test_simple_nl(self):
        self.paired_gather_test("simple","nl")

    # def test_simple_cuda(self):
    #     self.paired_gather_test("simple","cuda")

    # def test_nl_cuda(self):
    #     self.paired_gather_test("nl","cuda")

    def paired_gather_test(self,type_a,type_b):
        # compare with the simple "gather" function

        # ----------------------------
        #
        #     Setup Simple Test
        #
        # ----------------------------

        # -- set seed --
        seed = 123
        th.manual_seed(seed)
        np.random.seed(seed)

        # -- init save path --
        save_dir = SAVE_DIR
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # -- exec test 1 --
        sigma = 50.
        dname = "text_tourbus"
        args = edict({'ps':7,'pt':1,'k':10,'ws':10,'wt':5})

        # -- init vars --
        device = "cuda:0"
        clean_flow = True
        comp_flow = False

        # -- load data --
        vid = dnls.testing.data.load_burst("./data/",dname)
        vid = th.from_numpy(vid).to(device)
        # vid = vid + sigma * th.randn_like(vid)
        flow = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,sigma)

        # -- unpack params --
        k = args.k
        ps = args.ps
        pt = args.pt
        ws = args.ws
        wt = args.wt
        chnls = 1

        # -- batching info --
        device = vid.device
        shape = vid.shape
        ishape = vid.shape
        t,c,h,w = shape
        npix = t * h * w
        qStride = 1
        qSize = 100
        nsearch = (npix-1) // qStride + 1
        nbatches = (nsearch-1) // qSize + 1

        # -- gather funcs --
        gather_a = self.get_gather_func(type_a,vid.shape,device)
        gather_b = self.get_gather_func(type_b,vid.shape,device)

        # -- nbatches --
        for index in range(nbatches):

            # -- get [patches & nlInds] --
            queryInds = dnls.utils.inds.get_query_batch(index,qSize,qStride,
                                                        t,h,w,device)
            nlDists,nlInds = dnls.simple.search.run(vid,queryInds,
                                                    flow,k,ps,pt,ws,wt,chnls)
            patches = dnls.simple.scatter.run(vid,nlInds,ps,pt=1,dilation=1)
            nlInds = nlInds[:,[0],:] # only k = 1

            # -- testing forward --
            vid_nl_fwd,_ = gather_a(patches,nlDists,nlInds,shape=ishape)
            vid_simp_fwd,_ = gather_b(patches,nlDists,nlInds,shape=ishape)
            error = th.mean((vid_nl_fwd - vid_simp_fwd)**2).item()
            assert error < 1e-10

            # -- testing backward --
            vid_grad = th.ones_like(vid)
            vid_nl_bwd = autograd.backward(vid_grad)
            vid_simp_bwd = gather_simp.backward(patches,nlInds)
            error = th.mean((vid_nl_bwd - vid_simp_bwd)**2).item()
            assert error < 1e-10

    def test_gather_full(self,dname,sigma,comp_flow,args):

        # ----------------------------
        #
        #     Setup Simple Test
        #
        # ----------------------------

        # -- set seed --
        seed = 123
        th.manual_seed(seed)
        np.random.seed(seed)

        # -- init save path --
        save_dir = SAVE_DIR
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # -- exec test 1 --
        sigma = 50.
        dname = "text_tourbus"
        args = edict({'ps':7,'pt':1,'k':10,'ws':10,'wt':5})

        # -- init vars --
        device = "cuda:0"
        clean_flow = True
        comp_flow = False

        # -- load data --
        vid = dnls.testing.data.load_burst("./data/",dname)
        vid = th.from_numpy(vid).to(device)
        flow = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,sigma)

        # -- unpack params --
        k = args.k
        ps = args.ps
        pt = args.pt
        ws = args.ws
        wt = args.wt
        chnls = 1

        # -- batching info --
        device = vid.device
        ishape = vid.shape
        shape = vid.shape
        t,c,h,w = shape
        npix = t * h * w
        qStride = 1
        qSize = 100
        nsearch = (npix-1) // qStride + 1
        nbatches = (nsearch-1) // qSize + 1

        # -- nbatches --
        for index in range(nbatches):

            # -- get [patches & nlInds] --
            queryInds = dnls.utils.inds.get_query_batch(index,qSize,qStride,
                                                        t,h,w,device)
            nlDists,nlInds = dnls.simple.search.run(vid,queryInds,
                                                    flow,k,ps,pt,ws,wt,chnls)
            patches = dnls.simple.scatter.run(vid,nlInds,ps,pt=1,dilation=1)

            # -- exec gather fxns --
            gather_nl = dnls.gather.GatherNL(vid.shape)
            gather_simp = dnls.simple.gather.run

            # -- testing forward --
            vid_nl_fwd = gather_nl(patches,nlDists,nlInds,shape=ishape)
            vid_simp_fwd = gather_simp(patches,nlDists,nlInds,shape=ishape)
            error = th.mean((vid_nl_fwd - vid_simp_fwd)**2).item()
            assert error < 1e-10

            # -- testing backward --
            vid_grad = th.ones_like(vid)
            vid_nl_bwd = autograd.backward(vid_grad)
            vid_simp_bwd = gather_simp.backward(patches,nlInds)
            error = th.mean((vid_nl_bwd - vid_simp_bwd)**2).item()
            assert error < 1e-10



