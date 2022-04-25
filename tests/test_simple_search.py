
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

def run_rgb2gray(tensor):
    kernel = th.tensor([0.2989, 0.5870, 0.1140], dtype=th.float32)
    kernel = kernel.view(1, 3, 1, 1)
    rgb2gray = th.nn.Conv2d(in_channels=3,out_channels=1,kernel_size=(1, 1),bias=False)
    rgb2gray.weight.data = kernel
    rgb2gray.weight.requires_grad = False
    rgb2gray = rgb2gray.to(tensor.device)
    tensor = rgb2gray(tensor)
    return tensor


#
# -- Primary Testing Class --
#

class TestSimpleSearch(unittest.TestCase):

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
        npix_t = h * w
        qStride = 1
        qSearchTotal_t = npix_t // qStride # _not_ a DivUp
        qSearchTotal = t * qSearchTotal_t
        qSearch = qSearchTotal
        nbatches = (qSearchTotal - 1) // qSearch + 1

        # -- get patches with search --
        index = 0
        queryInds = dnls.utils.inds.get_query_batch(index,qSearch,qStride,t,h,w,device)
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
        error = th.max(((vid_ss - vid_uf)/255.)**2).item()
        assert error < 1e-10
        error = th.mean(((vid_ss - vid_uf)/255.)**2).item()
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
        npix_t = h * w
        qStride = 1
        qSearchTotal_t = npix_t // qStride # _not_ a DivUp
        qSearchTotal = t * qSearchTotal_t
        qSearch = qSearchTotal
        nbatches = (qSearchTotal - 1) // qSearch + 1

        # -- get patches with search --
        index = 0
        queryInds = dnls.utils.inds.get_query_batch(index,qSearch,qStride,t,h,w,device)
        nlDists,nlInds = dnls.simple.search.run(clean,queryInds,flow,k,
                                                ps,pt,ws,wt,chnls)
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
        npix_t = h * w
        qStride = 1
        qSearchTotal_t = npix_t // qStride # _not_ a DivUp
        qSearchTotal = t * qSearchTotal_t
        qSearch = qSearchTotal
        nbatches = (qSearchTotal - 1) // qSearch + 1

        # -- nbatches --
        for index in range(nbatches):

            # -- get [patches & nlInds] --
            queryInds = dnls.utils.inds.get_query_batch(index,qSearch,qStride,
                                                        t,h,w,device)
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

    def exec_matching_dists_test(self,dname,sigma,flow_args,args):
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
        npix_t = h * w
        qStride = 1
        qSearchTotal_t = npix_t // qStride # _not_ a DivUp
        qSearchTotal = t * qSearchTotal_t
        qSearch = qSearchTotal
        nbatches = (qSearchTotal - 1) // qSearch + 1

        # -- get patches with search --
        index = 0
        queryInds = dnls.utils.inds.get_query_batch(index,qSearch,qStride,t,h,w,device)
        nlDists,nlInds = dnls.simple.search.run(clean,queryInds,flow,k,
                                                ps,pt,ws,wt,3)
        patches = dnls.simple.scatter.run(clean,nlInds,ps,pt)/255.
        print("patches.shape: ",patches.shape)

        # -- unfold for comp --
        pad = ps//2
        clean_pad = pad_fxn(clean,(pad,pad,pad,pad),padding_mode="reflect")
        patches_uf = unfold(clean_pad,(ps,ps))/255.
        patches_uf = rearrange(patches_uf,'t d (h w) -> t h w d',h=h)
        print("patches_uf.shape: ",patches_uf.shape)

        p1_og = patches[-1,1].view(3,7,7)
        p0_og = patches[-1,0].view(3,7,7)
        p0 = patches_uf[9,31,31].view(3,7,7)
        # p0 = patches_uf[queryInds[-1,0],queryInds[-1,1],queryInds[-1,2]]
        # p1 = patches_uf[nlInds[-1,1,0],nlInds[-1,1,1],nlInds[-1,1,2]]
        p1 = patches_uf[3,31,30].view(3,7,7)
        print("p0.shape: ",p0.shape)
        dist = th.sum((p0 - p1)**2).item()
        print("Dist: ",dist)
        dist = th.sum((p0 - p0_og)**2).item()
        print("Dist[p0-og]: ",dist)
        dist = th.sum((p1 - p1_og)**2).item()
        print("Dist[p1-og]: ",dist)

        print("-"*20)
        print("-"*20)
        # print(clean[9,:,28,28]/255.)
        # print(clean[3,:,27,26]/255.)
        print(nlInds[-1])
        print(clean[9,:,27:,27:])
        print(patches[-1,0].view(3,7,7)*255.)
        print(clean[3,:,27:,26:])
        print(patches[-1,1].view(3,7,7)*255.)
        # print(clean[9,:,27,27]/255.)
        # print(clean[3,:,27,26]/255.)
        print("-"*20)
        print("-"*20)

        # print(p0)
        # print(p0_og)
        print(th.sum((p0 - p0_og)**2).item())
        print(th.sum((p0_og - p1_og)**2).item())

        # -- re-compute dists --
        np,k = patches.shape[:2]
        patches = patches.view(np,k,-1)
        pDists = th.sum((patches - patches[:,[0]])**2,dim=-1)

        print("pDists")
        print(pDists[-3:])
        print("nlDists")
        print(nlDists[-3:])

        # -- compute error -
        error = th.sum((pDists - nlDists)**2).item()
        print("error: ",error)
        assert error < 1e-10


    def manual_patch(img,inds):
        patches = img[9,32-2:,32-2:]
        zpatch = th.zeros((5,5),dtype=th.float32)
        zpatch[...] = patches[...]

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
        # self.exec_folding_test(dname,sigma,flow_args,args)
        # self.exec_topk_inds_test(dname,sigma,flow_args,args)
        # self.exec_nonincreasing_test(dname,sigma,flow_args,args)
        self.exec_matching_dists_test(dname,sigma,flow_args,args)
