
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

# -- testing --
from torch.nn.functional import nnf

# -- paths --
SAVE_DIR = Path("./output/tests/")

#
# -- Primary Testing Class --
#

class TestGather(unittest.TestCase):

    #
    # -- Primary Testing Loop --
    #

    def exec_simple_test(self,dname,sigma,comp_flow,args):
        # compare with the simple "gather" function

        # -- init vars --
        device = "cuda:0"
        clean_flow = True
        comp_flow = False

        # -- load data --
        clean = dnls.testing.data.load_burst(dname)
        clean = th.from_numpy(clean).to(device)
        noisy = clean + sigma * th.randn_like(clean)
        flow = dnls.testing.flow.get_flow(comp_flow,clean_flow,noisy,clean,sigma)

        # -- unpack params --
        k = args.k
        ps = args.ps
        pt = args.pt
        ws = args.ws
        wt = args.wt

        # -- batching info --
        device = noisy.device
        shape = noisy.shape
        t,c,h,w = shape
        npix = t * h * w
        qStride = 1
        qSize = 100
        nsearch = (npix-1) // qStride + 1
        nbatches = (nsearch-1) // qSize + 1

        # -- nbatches --
        for index in range(nbatches):

            # -- get [patches & nlInds] --
            queryInds = dnls.utils.get_query_batch(index,qSize,qStride,h,w,device)
            patches,nlDists,nlInds = dnls.simple.search.run(vid,queryInds,
                                                            flow,k,ps,pt,ws,wt)

            # -- exec gather fxns --
            gather_nl = dnls.gather.GatherNL(clean.shape)
            gather_simp = dnls.simple.gather.GatherSimple(clean.shape)

            # -- testing forward --
            vid_nl_fwd = gather_nl(patches,nlInds)
            vid_simp_fwd = gather_simp.forward(patches,nlInds)
            error = th.mean((vid_nl_fwd - vid_simp_fwd)**2).item()
            assert error < 1e-10

            # -- testing backward --
            vid_grad = th.ones_like(vid)
            vid_nl_bwd = autograd.backward(vid_grad)
            vid_simp_bwd = gather_simp.backward(patches,nlInds)
            error = th.mean((vid_nl_bwd - vid_simp_bwd)**2).item()
            assert error < 1e-10
    #
    # -- Launcher --
    #

    def test_gather(self):

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
        dname = "text_tourbus_64"
        args = edict({'ps':7,'pt':1,'k':10,'ws':10,'wt':5})
        self.exec_simple_test(dname,sigma,comp_flow,args)
