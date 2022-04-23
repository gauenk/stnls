
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
from torch.nn.functional import unfold,fold
from torchvision.transforms.functional import pad as pad_fxn
from torchvision.transforms.functional import center_crop

# -- Global Vars --
VIZ = True
SAVE_DIR = Path("./output/tests/")

#
# -- Primary Testing Class --
#

class TestSimpleGather(unittest.TestCase):

    #
    # -- Primary Testing Loop --
    #

    def exec_folding_test(self,dname,sigma,flow_args,args):
        """
        Check that "nearest neighbor" is the same as "unfold"
        """

        # -- load data --
        device = args.device
        clean = dnls.testing.data.load_burst("./data",dname)[:10]
        clean = clean[:,:,:32,:32]
        clean = th.from_numpy(clean).to(device)
        noisy = clean + sigma * th.randn_like(clean)
        flow = dnls.testing.flow.get_flow(flow_args.comp_flow,flow_args.clean_flow,
                                          noisy,clean,sigma)

        # -- unpack params --
        k = args.k
        ps = args.ps
        pt = args.pt
        ws = args.ws
        wt = args.wt
        chnls = args.chnls

        # -- batching info --
        device = noisy.device
        shape = noisy.shape
        t,c,h,w = shape
        npix = t * h * w
        qStride = 1
        qSize = npix
        nsearch = (npix-1) // qStride + 1
        nbatches = (nsearch-1) // qSize + 1

        # -- get patches with search --
        index = 0
        queryInds = dnls.utils.inds.get_query_batch(index,qSize,qStride,h,w,device)
        nlDists,nlInds = dnls.simple.search.run(clean,queryInds,
                                                flow,k,ps,pt,ws,wt,chnls)
        patches = dnls.simple.scatter.run(clean,nlInds,ps,pt)
        patches = rearrange(patches[:,0,0],'(t q) c h w -> t (c h w) q',t=t)

        # -- get patches with unfold --
        pad = ps//2
        clean_pad = pad_fxn(clean,(pad,pad,pad,pad),padding_mode="reflect")
        patches_uf = unfold(clean_pad,(ps,ps))

        # -- fold with k = 1 --
        hp,wp = h+2*pad,w+2*pad
        ones = th.ones_like(patches)
        Z = fold(ones,(hp,wp),(ps,ps))
        vid_ss = fold(patches,(hp,wp),(ps,ps)) / Z
        vid_uf = fold(patches_uf,(hp,wp),(ps,ps)) / Z

        # -- crop to center --
        vid_ss = center_crop(vid_ss,(h,w))
        vid_uf = center_crop(vid_uf,(h,w))

        # -- visualize --
        if VIZ:
            delta = th.abs(vid_ss - vid_uf)# / 255.
            dnls.testing.data.save_burst(clean,SAVE_DIR,"clean")
            dnls.testing.data.save_burst(vid_ss,SAVE_DIR,"vid_ss")
            dnls.testing.data.save_burst(vid_uf,SAVE_DIR,"vid_uf")
            dnls.testing.data.save_burst(delta,SAVE_DIR,"delta")

        # -- testing --
        error = th.mean((vid_ss - vid_uf)**2).item()
        assert error < 1e-10
        error = th.max((vid_ss - vid_uf)**2).item()
        assert error < 1e-10


    def exec_topk_inds_test(self,dname,sigma,flow_args,args):
        """
        Check that the "1st nearest neighbor" is the same queryInds
        """

        # -- load data --
        device = args.device
        clean = dnls.testing.data.load_burst("./data",dname)[:10]
        clean = clean[:,:,:32,:32]
        clean = th.from_numpy(clean).to(device)
        noisy = clean + sigma * th.randn_like(clean)
        flow = dnls.testing.flow.get_flow(flow_args.comp_flow,flow_args.clean_flow,
                                          noisy,clean,sigma)

        # -- unpack params --
        k = args.k
        ps = args.ps
        pt = args.pt
        ws = args.ws
        wt = args.wt
        chnls = args.chnls

        # -- batching info --
        device = noisy.device
        shape = noisy.shape
        t,c,h,w = shape
        npix = t * h * w
        qStride = 1
        qSize = npix
        nsearch = (npix-1) // qStride + 1
        nbatches = (nsearch-1) // qSize + 1

        # -- get patches with search --
        index = 0
        queryInds = dnls.utils.inds.get_query_batch(index,qSize,qStride,h,w,device)
        nlDists,nlInds = dnls.simple.search.run(clean,queryInds,
                                                flow,k,ps,pt,ws,wt,chnls)
        patches = dnls.simple.scatter.run(clean,nlInds,ps,pt)

        # -- test topk index --
        dinds = th.sum((nlInds[:,0] - queryInds)**2).item()
        assert dinds == 0,"nearest neighbors is query index."


    def exec_nonincreasing_test(self,dname,sigma,flow_args,args):
        """
        Check that nearest neighbors have a non-increasing nosiy patch differences
        """

        # -- load data --
        device = args.device
        clean = dnls.testing.data.load_burst("./data",dname)[:2]
        clean = th.from_numpy(clean).to(device)
        noisy = clean + sigma * th.randn_like(clean)
        flow = dnls.testing.flow.get_flow(flow_args.comp_flow,flow_args.clean_flow,
                                          noisy,clean,sigma)

        # -- unpack params --
        k = args.k
        ps = args.ps
        pt = args.pt
        ws = args.ws
        wt = args.wt
        chnls = args.chnls

        # -- batcing info --
        device = noisy.device
        shape = noisy.shape
        t,c,h,w = shape
        npix = t * h * w
        qStride = 1
        qSize = npix
        nsearch = (npix-1) // qStride + 1
        nbatches = (nsearch-1) // qSize + 1

        # -- nbatches --
        for index in range(nbatches):

            # -- get [patches & nlInds] --
            queryInds = dnls.utils.inds.get_query_batch(index,qSize,qStride,h,w,device)
            nlDists,nlInds = dnls.simple.search.run(clean,queryInds,
                                                    flow,k,ps,pt,ws,wt,chnls)
            patches = dnls.simple.scatter.run(clean,nlInds,ps,pt)

            # -- torch mean --
            patches = rearrange(patches,'q k t c h w -> q k (t c h w)')
            dpatches = th.mean((patches - patches[:,[0]])**2,-1)

            # -- sort across "k" --
            for i in range(k-1):
                ineq = dpatches[:,i] <= dpatches[:,i+1]
                ineq = th.all(ineq).item()
                assert ineq

    #
    # -- Launcher --
    #

    def test_simple_search(self):

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
        device = 'cuda:0'
        comp_flow = False
        dname = "text_bus"
        args = edict({'ps':7,'pt':1,'k':3,'ws':10,'wt':5,'chnls':3,'device':device})
        flow_args = edict({'comp_flow':False,'clean_flow':False})
        self.exec_folding_test(dname,sigma,flow_args,args)
        self.exec_topk_inds_test(dname,sigma,flow_args,args)
        self.exec_nonincreasing_test(dname,sigma,flow_args,args)

