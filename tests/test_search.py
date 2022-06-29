
# -- python --
import cv2,tqdm,copy,pytest
import numpy as np
import unittest
import tempfile
import sys
import shutil
from pathlib import Path
from easydict import EasyDict as edict

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- package helper imports --
import dnls

# -- check if reordered --
from scipy import optimize
SAVE_DIR = Path("./output/tests/")

#
#
# -- Primary Testing Class --
#
#

class TestTopKSearch(unittest.TestCase):

    #
    # -- Load Data --
    #

    def do_load_data(self,dname,sigma,device="cuda:0"):

        #  -- Read Data (Image & VNLB-C++ Results) --
        ext = "jpg"
        vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
        clean = th.from_numpy(vid).to(device).contiguous()
        clean = clean * 1.0
        noisy = clean + sigma * th.normal(0,1,size=clean.shape,device=device)
        return clean,noisy

    def do_load_flow(self,comp_flow,burst,sigma,device):
        #  -- Empty shells --
        t,c,h,w = burst.shape
        tf32,tfl = th.float32,th.long
        fflow = th.zeros(t,2,h,w,dtype=tf32,device=device)
        bflow = fflow.clone()

        # -- pack --
        flows = edict()
        flows.fflow = fflow
        flows.bflow = bflow
        return flows


    def init_topk_shells(self,bsize,k,device):
        tf32,ti32 = th.float32,th.int32
        vals = float("inf") * th.ones((bsize,k),dtype=tf32,device=device)
        inds = -th.ones((bsize,k),dtype=ti32,device=device)
        return vals,inds

    #
    # -- [Exec] Sim Search --
    #

    def run_comparison(self,dname,sigma,args):

        # -- get data --
        noisy,clean = self.do_load_data(dname,sigma)

        # -- fixed testing params --
        k = 15
        BSIZE = 50
        NBATCHES = 3
        shape = noisy.shape
        device = noisy.device
        t,c,h,w = noisy.shape
        npix = h*w

        # -- create empty bufs --
        bufs = edict()
        bufs.patches = None
        bufs.dists = None
        bufs.inds = None

        # -- batching info --
        device = noisy.device
        shape = noisy.shape
        t,c,h,w = shape
        npix_t = h * w
        qStride = 1
        qSearchTotal_t = npix_t // qStride # _not_ a DivUp
        qSearchTotal = t * qSearchTotal_t
        qSearch = qSearchTotal
        nbatches = (qSearchTotal - 1) // qSearch + 1

        # -- unpack --
        ps = args.ps
        pt = args.pt
        ws = args.ws
        wt = args.wt
        chnls = args.chnls

        # -- flows --
        comp_flow = True
        clean_flow = True
        flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,
                                           noisy,clean,sigma)

        # -- final args --
        args.c = c
        args['stype'] = "faiss"
        args['queryStride'] = 7
        args['bstride'] = args['queryStride']

        # -- exec over batches --
        for index in range(NBATCHES):

            # -- new image --
            clean = th.rand_like(clean).type(th.float32)

            # -- queries --
            index = 0
            queryInds = dnls.utils.inds.get_query_batch(index,qSearch,qStride,
                                                        t,h,w,device)

            # -- search using python code --
            nlDists_simp,nlInds_simp = dnls.simple.search.run(clean,queryInds,
                                                              flows,k,ps,pt,ws,wt,chnls)

            # -- search using CUDA code --
            dnls_search = dnls.search.SearchNl(flows.fflow, flows.bflow, k, ps, pt,
                                               ws, wt, chnls=chnls,dilation=1, stride=1)
            nlDists_cu,nlInds_cu = dnls_search(clean,queryInds)

            # -- to numpy --
            nlDists_cu = nlDists_cu.cpu().numpy()
            nlDists_simp = nlDists_simp.cpu().numpy()
            nlInds_cu = nlInds_cu.cpu().numpy()
            nlInds_simp = nlInds_simp.cpu().numpy()

            # -- save mask --
            dists_cu = rearrange(nlDists_cu,'(t h w) k -> t k h w ',t=t,h=h,w=w)
            dists_simp = rearrange(nlDists_simp,'(t h w) k -> t k h w ',t=t,h=h,w=w)
            dists = np.abs(dists_cu - dists_simp)
            for ti in range(t):
                dists_ti = repeat(dists[ti,:,:,:],'t h w -> t c h w ',c=3)
                if dists_ti.max() > 1e-3: dists_ti /= dists_ti.max()
                dnls.testing.data.save_burst(dists_ti,SAVE_DIR,"dists_%d" % ti)

            # -- allow for swapping of "close" values --
            np.testing.assert_array_almost_equal(nlDists_cu,nlDists_simp,5)

            # -- mostly the same inds --
            perc_neq = (np.abs(nlInds_cu != nlInds_simp)*1.).mean().item()
            assert perc_neq < 0.05

    def test_sim_search(self):

        # -- init save path --
        np.random.seed(123)
        save_dir = SAVE_DIR
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # -- test 1 --
        sigma = 25.
        dname = "davis_baseball_64x64"
        # dname = "text_tourbus_64"
        args = edict({'ps':7,'pt':1,"ws":10,"wt":10,"chnls":1})
        nreps = 3
        for r in range(nreps):
            self.run_comparison(dname,sigma,args)

    # @pytest.mark.skip()
    def test_sim_search_fwd_bwd(self):

        # -- init save path --
        np.random.seed(123)
        save_dir = SAVE_DIR
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # -- test 1 --
        sigma = 25.
        # dname = "text_tourbus_64"
        dname = "davis_baseball_64x64"

        # -- get data --
        noisy,clean = self.do_load_data(dname,sigma)

        # -- fixed testing params --
        k = 15
        BSIZE = 50
        NBATCHES = 3
        shape = noisy.shape
        device = noisy.device
        t,c,h,w = noisy.shape
        npix = h*w

        # -- create empty bufs --
        bufs = edict()
        bufs.patches = None
        bufs.dists = None
        bufs.inds = None

        # -- batching info --
        device = noisy.device
        shape = noisy.shape
        t,c,h,w = shape
        npix_t = h * w
        qStride = 1
        qSearchTotal_t = npix_t // qStride # _not_ a DivUp
        qSearchTotal = t * qSearchTotal_t
        qSearch = qSearchTotal
        nbatches = (qSearchTotal - 1) // qSearch + 1

        # -- unpack --
        ps = 5
        pt = 1
        ws = 5
        wt = 2
        chnls = 1

        # -- flows --
        comp_flow = True
        clean_flow = True
        flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,
                                           noisy,clean,sigma)

        # -- new image --
        clean = th.rand_like(clean).type(th.float32)
        clean.requires_grad_(True)

        # -- queries --
        index = 0
        queryInds = dnls.utils.inds.get_query_batch(index,qSearch,qStride,
                                                    t,h,w,device)

        # -- search using CUDA code --
        dnls_search = dnls.search.SearchNl(flows.fflow, flows.bflow, k, ps, pt,
                                           ws, wt, dilation=1, stride=1)
        nlDists,nlInds = dnls_search(clean,queryInds)
        ones = th.rand_like(nlDists)
        loss = th.sum((nlDists - ones)**2)
        loss.backward()


