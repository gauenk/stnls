
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

class TestScatter(unittest.TestCase):

    #
    # -- Primary Testing Loop --
    #

    def exec_simple_test(self,dname,sigma,comp_flow,args):
        # compare with the simple "scatter" function

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
            nlDists,nlInds = dnls.simple.search.run(vid,queryInds,
                                                    flow,k,ps,pt,ws,wt)

            # -- exec scatter fxns --
            scatter_nl = dnls.scatter.ScatterNL(ps,pt)
            scatter_simp = dnls.simple.scatter.ScatterSimple(ps,pt)

            # -- testing forward --
            patches_nl_fwd = scatter_nl(vid_wg,nlInds)
            patches_simp_fwd = scatter_simp.forward(vid,nlInds)
            error = th.mean((patches_nl_fwd - patches_simp_fwd)**2).item()
            assert error < 1e-10

            # -- testing backward --
            patches_grad = th.randn_like(patches)
            patches_nl_bwd = autograd.backward(patches,patches_grad)
            patches_simp_bwd = scatter_simp.backward(patches,nlInds)
            error = th.mean((patches_nl_bwd - patches_simp_bwd)**2).item()
            assert error < 1e-10
    #
    # -- Launcher --
    #

    def test_scatter(self):

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
