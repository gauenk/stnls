
# -- python --
import cv2,tqdm,copy
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
import vpss
import vnlb
from faiss.contrib import kn3
from faiss.contrib import testing

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
        clean = testing.load_dataset(dname).to(device)[:5]
        clean = th.zeros((15,3,32,32)).to(device)
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

    def get_search_inds(self,index,bsize,stride,shape,device):
        t,c,h,w  = shape
        start = index * bsize
        stop = ( index + 1 ) * bsize
        ti32 = th.int32
        srch_inds = th.arange(start,stop,stride,dtype=ti32,device=device)[:,None]
        srch_inds = kn3.get_3d_inds(srch_inds,h,w)
        srch_inds = srch_inds.contiguous()
        return srch_inds

    def init_topk_shells(self,bsize,k,device):
        tf32,ti32 = th.float32,th.int32
        vals = float("inf") * th.ones((bsize,k),dtype=tf32,device=device)
        inds = -th.ones((bsize,k),dtype=ti32,device=device)
        return vals,inds

    def exec_vpss_search(self,K,clean,flows,sigma,args):
        vpss_mode = args['vpss_mode']
        if vpss_mode == "exh":
            return self.exec_vpss_search_exh(K,clean,flows,sigma,args)
        elif vpss_mode == "vnlb":
            return self.exec_vpss_search_vnlb(K,clean,flows,sigma,args)
        else:
            raise ValueError(f"Uknown vpss_mode [{vpss_mode}]")

    def exec_vpss_search_vnlb(self,K,clean,flows,sigma,args):

        # -- vnlb --
        bufs = vnlb.global_search_default(clean,sigma,None,args.ps,K,pfill=True)
        vpss_patches = bufs.patches

        # -- return --
        th.cuda.synchronize()

        # -- weight floating-point issue --
        vpss_patches = vpss_patches.type(th.float32)

        return vpss_patches

    def exec_vpss_search_exh(self,K,clean,flows,sigma,args):

        # -- unpack --
        device = clean.device
        shape = clean.shape
        t,c,h,w = shape

        # -- get search inds --
        index,BSIZE,stride = 0,t*h*w,args.bstride
        srch_inds = self.get_search_inds(index,BSIZE,stride,shape,device)
        srch_inds = srch_inds.type(th.int32)

        # -- get return shells --
        numQueries = ((BSIZE - 1)//args.bstride)+1
        pt,c,ps = args.pt,args.c,args.ps
        vpss_vals,vpss_inds = self.init_topk_shells(numQueries,K,device)

        # -- search using numba code --
        vpss.exec_sim_search_burst(clean,srch_inds,vpss_vals,vpss_inds,flows,sigma,args)

        # # -- fill patches --
        # vpss.get_patches_burst(clean,vpss_inds,ps,pt=pt,patches=None,fill_mode="faiss")

        # -- return --
        th.cuda.synchronize()

        return vpss_vals,vpss_inds,srch_inds

    def exec_kn3_search(self,K,clean,flows,sigma,args,bufs):

        # -- unpack --
        device = clean.device
        shape = clean.shape
        t,c,h,w = shape

        # -- init inputs --
        index,BSIZE = 0,t*h*w
        args.k = K
        numQueries = (BSIZE-1) // args.queryStride + 1

        # -- search --
        kn3.run_search(clean,0,numQueries,flows,sigma,args,bufs)
        th.cuda.synchronize()

        kn3_vals = bufs.dists
        kn3_inds = bufs.inds

        return kn3_vals,kn3_inds


    #
    # -- [Exec] Sim Search --
    #

    def run_vpss_comparison(self,noisy,clean,sigma,flows,args):

        # -- fixed testing params --
        K = 15
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

        # -- final args --
        args.c = c
        args['stype'] = "faiss"
        args['vpss_mode'] = "exh"
        args['queryStride'] = 7
        args['bstride'] = args['queryStride']
        # args['vpss_mode'] = "vnlb"

        # -- exec over batches --
        for index in range(NBATCHES):

            # -- new image --
            clean = 255.*th.rand_like(clean).type(th.float32)

            # -- search using python code --
            vpss_vals,vpss_inds,srch_inds = self.exec_vpss_search(K,clean,flows,
                                                                  sigma,args)

            # -- search using faiss code --
            kn3_vals,kn3_inds = self.exec_kn3_search(K,clean,flows,sigma,args,bufs)

            # -- to numpy --
            vpss_vals = vpss_vals.cpu().numpy()
            kn3_vals = kn3_vals.cpu().numpy()


            neq = np.where(np.abs(kn3_vals - vpss_vals)>1)
            if len(neq[0]) > 0:
                bidx = neq[0][0]
                print(bidx)
                print(srch_inds.shape)
                print(vpss_vals.shape)
                print(srch_inds[-3:])
                print(srch_inds[bidx])
                bt,bh,bw = (bidx // npix),(bidx // npix)//w,(bidx // npix)%w
                print(bt,bh,bw)
                print(neq)
                print(kn3_vals[neq])
                print(vpss_vals[neq])
                print(vpss_inds[bidx,0])

            # -- allow for swapping of "close" values --
            np.testing.assert_array_almost_equal(kn3_vals,vpss_vals)

    def run_single_vpss_test(self,dname,sigma,comp_flow,pyargs):
        noisy,clean = self.do_load_data(dname,sigma)
        flows = self.do_load_flow(False,clean,sigma,noisy.device)
        self.run_vpss_comparison(noisy,clean,sigma,flows,pyargs)

    # def test_vpss_sim_search(self):

    #     # -- init save path --
    #     np.random.seed(123)
    #     save_dir = SAVE_DIR
    #     if not save_dir.exists():
    #         save_dir.mkdir(parents=True)

    #     # -- test 1 --
    #     sigma = 25.
    #     dname = "text_tourbus_64"
    #     comp_flow = False
    #     args = edict({'ps':7,'pt':1})
    #     self.run_single_vpss_test(dname,sigma,comp_flow,args)

    #
    # -- [Exec] Sim Search --
    #

    def run_comparison(self,noisy,clean,sigma,args):

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
        args['vpss_mode'] = "exh"
        args['queryStride'] = 7
        args['bstride'] = args['queryStride']
        # args['vpss_mode'] = "vnlb"

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
                                               ws, wt, dilation=1, stride=1)
            nlDists_cu,nlInds_cu = dnls_search(clean,queryInds)

            # -- to numpy --
            nlDists_cu = nlDists_cu.cpu().numpy()
            nlDists_simp = nlDists_simp.cpu().numpy()
            nlInds_cu = nlInds_cu.cpu().numpy()
            nlInds_simp = nlInds_simp.cpu().numpy()

            # -- save mask --
            dists = (nlDists_cu - nlDists_simp)**2
            print(t,h,w)
            dists = rearrange(dists,'(t h w) k -> t k h w ',t=t,h=h,w=w)
            dists = repeat(dists[0,:,:,:],'t h w -> t c h w ',c=3)
            if dists.max() > 1e-3: dists /= dists.max()
            dnls.testing.data.save_burst(dists,SAVE_DIR,"dists")
            print(nlDists_cu.shape)
            print(nlDists_cu[0,-1],nlDists_simp[0,-1])
            print(nlDists_cu[0,10],nlDists_simp[0,10])
            print(nlDists_cu[0,-5:],nlDists_simp[0,-5:])

            # -- allow for swapping of "close" values --
            np.testing.assert_array_almost_equal(nlDists_cu,nlDists_simp,5)
            np.testing.assert_array_almost_equal(nlInds_cu,nlInds_simp)


    def run_single_test(self,dname,sigma,pyargs):
        noisy,clean = self.do_load_data(dname,sigma)
        self.run_comparison(noisy,clean,sigma,pyargs)

    def test_sim_search(self):

        # -- init save path --
        np.random.seed(123)
        save_dir = SAVE_DIR
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # -- test 1 --
        sigma = 25.
        dname = "text_tourbus_64"
        args = edict({'ps':7,'pt':1,"ws":10,"wt":10,"chnls":1})
        self.run_single_test(dname,sigma,args)

