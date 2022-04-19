
# -- python --
import cv2,tqdm,copy
import numpy as np
import unittest
import tempfile
import sys
from einops import rearrange
import shutil
from pathlib import Path
from easydict import EasyDict as edict

# -- vision --
from PIL import Image

# -- linalg --
import torch as th
import numpy as np

# -- package helper imports --
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
PYTEST_OUTPUT = Path("./pytests/output/")

def save_image(burst,prefix="prefix"):
    root = PYTEST_OUTPUT
    if not(root.exists()): root.mkdir()
    burst = rearrange(burst,'t c h w -> t h w c')
    burst = np.clip(burst,0,255)
    burst = burst.astype(np.uint8)
    nframes = burst.shape[0]
    for t in range(nframes):
        fn = "%s_kn3_io_%02d.png" % (prefix,t)
        img = Image.fromarray(burst[t])
        path = str(root / fn)
        img.save(path)

class TestIoPatches(unittest.TestCase):

    #
    # -- Load Data --
    #

    def do_load_data(self,dname,sigma,device="cuda:0"):

        #  -- Read Data (Image & VNLB-C++ Results) --
        clean = testing.load_dataset(dname)
        clean = clean[:15,:,:32,:32].to(device).type(th.float32)
        # clean = th.zeros((15,3,32,32)).to(device).type(th.float32)
        clean = clean * 1.0
        noisy = clean + sigma * th.normal(0,1,size=clean.shape,device=device)
        return clean,noisy

    def do_load_flow(self,comp_flow,burst,sigma,device):
        if comp_flow:
            #  -- TV-L1 Optical Flow --
            flow_params = {"nproc":0,"tau":0.25,"lambda":0.2,"theta":0.3,
                           "nscales":100,"fscale":1,"zfactor":0.5,"nwarps":5,
                           "epsilon":0.01,"verbose":False,"testing":False,'bw':True}
            fflow,bflow = vnlb.swig.runPyFlow(burst,sigma,flow_params)
        else:
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

    def get_search_inds(self,index,bsize,shape,device):
        t,c,h,w  = shape
        start = index * bsize
        stop = ( index + 1 ) * bsize
        ti32 = th.int32
        srch_inds = th.arange(start,stop,dtype=ti32,device=device)[:,None]
        srch_inds = kn3.get_3d_inds(srch_inds,h,w)
        srch_inds = srch_inds.contiguous()
        return srch_inds

    def init_topk_shells(self,bsize,k,pt,c,ps,device):
        tf32,ti32 = th.float32,th.int32
        vals = float("inf") * th.ones((bsize,k),dtype=tf32,device=device)
        inds = -th.ones((bsize,k),dtype=ti32,device=device)
        patches = -th.ones((bsize,k,pt,c,ps,ps),dtype=tf32,device=device)
        return vals,inds,patches

    def exec_kn3_search(self,K,clean,flows,sigma,args,bufs):

        # -- unpack --
        device = clean.device
        shape = clean.shape
        t,c,h,w = shape

        # -- prepare kn3 search  --
        index,BSIZE = 0,t*h*w
        args.k = K
        numQueries = (BSIZE-1) // args.queryStride + 1

        # -- search --
        kn3.run_search(clean,0,numQueries,flows,sigma,args,bufs,pfill=True)
        th.cuda.synchronize()

        # -- unpack --
        kn3_vals = bufs.dists
        kn3_inds = bufs.inds
        kn3_patches = bufs.patches

        return kn3_inds,kn3_patches

    #
    # -- [Exec] Sim Search --
    #

    def run_comparison_fill_p2b(self,noisy,clean,sigma,flows,args):

        # -- fixed testing params --
        K = 100 # problem one
        BSIZE = 50
        NBATCHES = 3
        shape = noisy.shape
        device = noisy.device

        # -- create empty bufs --
        bufs = edict()
        bufs.patches = None
        bufs.dists = None
        bufs.inds = None
        clean /= 255.
        clean *= 255.
        args['queryStride'] = 7
        args['stype'] = "faiss"

        # -- exec over batches --
        for index in range(NBATCHES):

            # -- get new image --
            noise = sigma * th.randn_like(clean)
            noisy = (clean + noise).type(th.float32).contiguous()
            # clean = 255.*th.rand_like(clean).type(th.float32)
            fill_img = -th.ones_like(clean).contiguous()

            # -- search using faiss code --
            bufs = edict()
            bufs.patches = None
            bufs.dists = None
            bufs.inds = None
            _,patches = self.exec_kn3_search(K,noisy,flows,sigma,args,bufs)

            # -- fill patches --
            kn3.run_fill(fill_img,patches,0,args,"p2b",clock=None)
            fmin,fmax = fill_img.min().item(),fill_img.max().item()

            # -- cpu --
            fill_img_np = fill_img.cpu().numpy()
            noisy_np = noisy.cpu().numpy()
            delta = 255.*(th.abs(fill_img - noisy) > 1e-6)
            delta_np = delta.cpu().numpy()
            save_image(fill_img_np,prefix="fill")
            save_image(noisy_np,prefix="clean")
            save_image(delta_np,prefix="delta")

            # -- test --
            np.testing.assert_array_equal(fill_img_np,noisy_np)

    def run_comparison_fill_b2p(self,noisy,clean,sigma,flows,args):

        # -- fixed testing params --
        K = 100
        BSIZE = 50
        NBATCHES = 3
        shape = noisy.shape
        device = noisy.device

        # -- create empty bufs --
        bufs = edict()
        bufs.patches = None
        bufs.dists = None
        bufs.inds = None
        clean /= 255.
        clean *= 255.
        args['queryStride'] = 7
        args['stype'] = "faiss"

        # -- exec over batches --
        for index in range(NBATCHES):

            # -- get new image --
            noise = sigma * th.randn_like(clean)
            noisy = (clean + noise).type(th.float32).contiguous()
            # clean = 255.*th.rand_like(clean).type(th.float32)
            fill_img = -th.ones_like(clean).contiguous()

            # -- search using faiss code --
            inds,patches = self.exec_kn3_search(K,noisy,flows,sigma,args,bufs)

            # -- fill patches --
            fpatches = th.zeros_like(patches)
            kn3.run_fill(noisy,fpatches,0,args,"b2p",inds=inds,clock=None)

            # -- cpu --
            patches_np = patches.cpu().numpy()
            fpatches_np = fpatches.cpu().numpy()

            # -- test --
            np.testing.assert_array_equal(patches_np,fpatches_np)

    def run_large_p2b(self,noisy,clean,sigma,flows,args):
        # -- fixed testing params --
        K = 100
        BSIZE = 50
        NBATCHES = 3
        shape = noisy.shape
        device = noisy.device

        # -- create empty bufs --
        bufs = edict()
        bufs.patches = None
        bufs.dists = None
        bufs.inds = None
        args['queryStride'] = 7
        args['stype'] = "faiss"
        noisy = th.zeros((10,3,128,128)).to(noisy.device)

        # -- exec over batches --
        for index in range(NBATCHES):

            # -- random patches --
            patches = th.rand((1024*20,26,1,3,5,5))

            # -- fill patches --
            kn3.run_fill(noisy,patches,0,args,"p2b")

        assert True

    def run_single_test(self,dname,sigma,comp_flow,pyargs):
        noisy,clean = self.do_load_data(dname,sigma)
        flows = self.do_load_flow(False,clean,sigma,noisy.device)
        self.run_comparison_fill_p2b(noisy,clean,sigma,flows,pyargs)
        self.run_comparison_fill_b2p(noisy,clean,sigma,flows,pyargs)
        self.run_large_p2b(noisy,clean,sigma,flows,pyargs)

    def test_sim_search(self):

        # -- init save path --
        np.random.seed(123)
        save_dir = SAVE_DIR
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # -- test 1 --
        sigma = 50./255.
        dname = "text_tourbus_64"
        comp_flow = False
        args = edict({'ps':7,'pt':1,'c':3})
        self.run_single_test(dname,sigma,comp_flow,args)
