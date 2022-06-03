
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

class TestiUnfold(unittest.TestCase):

    #
    # -- Test v.s. NN --
    #

    def test_nn_iunfold(self):

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
        stride = args.stride

        # -- batching info --
        device = noisy.device
        shape = noisy.shape
        t,c,h,w = shape
        npix = t * h * w
        nh = (h-1)//stride+1
        nw = (w-1)//stride+1
        qTotal = t * nh * nw
        qSize = qTotal
        nbatches = (qTotal-1) // qSize + 1
        vid = vid.contiguous()

        # -- exec iunfold fxns --
        coords = (2,2,h-2,w-2)
        coords = [2,2,h-6,w-12]
        coords = [2,4,h-6,w-12]
        # coords = (12,12,h-12,w-12)
        scatter_nl = dnls.scatter.ScatterNl(ps,pt,dilation=dil,exact=True)
        iunfold_nl = dnls.iunfold.iUnfold(ps,coords,stride=stride,dilation=dil)

        # -- get check bounds --
        psHalf = ps//2
        a,b,c,d = coords # top,left,btm,right
        a = (a+psHalf-1)//stride+1
        b = (b+psHalf-1)//stride+1
        c = (c-psHalf-1)//stride+1
        d = (d-psHalf-1)//stride+1


        #
        # -- test logic --
        #

        # -- prepare videos --
        psHalf = ps//2
        padf = psHalf*dil
        # vid = pad(vid,4*[padf,],mode="reflect")
        vid_nn = vid.clone()
        vid_nl = vid.clone()
        vid_nn.requires_grad_(True)
        vid_nl.requires_grad_(True)

        # -- prepare video with boarder --
        top,left,btm,right = coords
        hp = coords[2] - coords[0]
        wp = coords[3] - coords[1]
        padf = (h-hp)//2
        padp = dil*(ps//2)
        pad_tot = padf + padp
        vid_nn_cc = vid_nn[:,:,top:btm,left:right].contiguous()
        vid_nn_mod = pad(vid_nn_cc,[padp,]*4,mode="reflect")
        padfs = [left,w-right,top,h-btm]
        vid_nn_mod = pad(vid_nn_mod,padfs,mode="constant",value=0.)
        # print("[2] vid_nn_mod.shape: ",vid_nn_mod.shape)
        # print("vid_nn.shape: ",vid_nn.shape)
        # vid_nn_mod = pad(vid_nn,[padp,]*4,mode="reflect")
        # vid_nn_mod = vid_nn_mod

        # -- run forward --
        # print("noisy.shape: ",noisy.shape)
        patches_nn = self.run_unfold(vid_nn_mod,ps,stride,dil)
        patches_nl = iunfold_nl(vid_nl,0,qTotal)
        # print("patches_nn.shape: ",patches_nn.shape)
        # print("patches_nl.shape: ",patches_nl.shape)

        # -- save ex --
        viz_nn,w_nn = self.run_fold(patches_nn,t,h,w,stride,dil)
        viz_nl,w_nl = self.run_fold(patches_nl,t,h,w,stride,dil)
        viz_nn /= w_nn
        viz_nl /= w_nn
        # print("all zero? ",th.all(viz_nl == 0).item())
        # print(w_nn[0,0,:3,:3])
        # print(w_nl[0,0,:3,:3])
        # print(w_nn[0,0,-3:,-3:])
        # print(w_nl[0,0,-3:,-3:])
        # print("-"*20)
        # print("-"*20)
        # print(viz_nn[0,0,:4,:4])
        # print(viz_nl[0,0,:4,:4])
        # print("-"*20)
        # print(viz_nn[0,0,-4:,-4:])
        # print(viz_nl[0,0,-4:,-4:])
        # print("-"*20)
        dnls.testing.data.save_burst(viz_nn,SAVE_DIR,"vid_nn")
        dnls.testing.data.save_burst(viz_nl,SAVE_DIR,"vid_nl")

        # -- inspect --
        # print("-"*20)
        # print(vid_nn_cc[0,0,:3,:3])
        # print(vid_nn_mod[0,0,:3,:3])
        # print(vid_nl[0,0,:6,:6])
        # print("-"*20)
        # print(patches_nn[0,0,0,0])
        # print(patches_nl[0,0,0,0])
        # print("-"*20)
        # print(patches_nn[64//stride+1,0,0,0])
        # print(patches_nl[64//stride+1,0,0,0])
        # print("-"*20)
        # print(patches_nn[3*(64//stride)+10,0,0,0])
        # print(patches_nl[3*(64//stride)+10,0,0,0])
        # print("-"*20)
        # mid = (64//stride)*(32//stride)+32//stride
        # print(patches_nn[mid,0,0,0])
        # print(patches_nl[mid,0,0,0])
        # print("-"*20)
        # print(patches_nn[-1,0,0,0])
        # print(patches_nl[-1,0,0,0])
        # print("-"*20)

        # -- run backward --
        # a,b,c,d = a,b,c-psHalf,d-psHalf,
        # t,c,h,w = noisy.shape
        shape_str = '(t h w) 1 1 c ph pw -> t h w c ph pw'
        patches_nn = rearrange(patches_nn,shape_str,t=t,h=nh)
        patches_nl = rearrange(patches_nl,shape_str,t=t,h=nh)
        patches_grad = th.rand_like(patches_nn).type(th.float32)
        # print(patches_nn[0,0,0,0])
        # print(patches_nl[0,0,0,0])
        # print(patches_nn[0,16,16,0])
        # print(patches_nl[0,16,16,0])

        # print("-"*10)
        # for i in range(5):
        #     print("-"*10)
        #     print(patches_nn[0,i,i,0])
        #     print(patches_nl[0,i,i,0])

        # print("-"*10)
        # for i in range(5):
        #     print("-"*10)
        #     print(patches_nn[0,-i,-i,0])
        #     print(patches_nl[0,-i,-i,0])

        th.autograd.backward(patches_nn[:,a:c,b:d],patches_grad[:,a:c,b:d])
        th.autograd.backward(patches_nl[:,a:c,b:d],patches_grad[:,a:c,b:d])

        # -- diff map --
        # emap = th.mean((patches_nn - patches_nl)**2,(-1,-2))
        # print(a,b,c,d)
        # print("emap.shape: ",emap.shape)
        # # emap = emap[:,a:c,b:d]
        # print("emap.max(): ",emap.max().item())
        # emap /= emap.max().item()
        # emap = rearrange(emap,'t h w c -> t c h w')
        # dnls.testing.data.save_burst(emap,SAVE_DIR,"emap")

        # -- check forward --
        error = th.sum((patches_nn[:,a:c,b:d] - patches_nl[:,a:c,b:d])**2).item()
        assert error < 1e-10

        # -- check backward --
        hp = coords[2] - coords[0]
        wp = coords[3] - coords[1]
        grad_nn = vid_nn.grad[:,:,top:btm,left:right]
        grad_nl = vid_nl.grad[:,:,top:btm,left:right]

        # -- vis --
        # print("\n")
        # print(grad_nn[0,0,:3,:3])
        # print(grad_nl[0,0,:3,:3])
        # print("-"*20)
        # print(grad_nn[0,0,-3:,-3:])
        # print(grad_nl[0,0,-3:,-3:])
        # print("-"*20)
        # print(grad_nn[0,0,8:14,8:14])
        # print(grad_nl[0,0,8:14,8:14])

        # -- compute error --
        diff = th.abs(grad_nn - grad_nl)
        dmax = diff.max()
        if dmax > 1e-3: diff /= dmax
        dnls.testing.data.save_burst(diff,SAVE_DIR,"diff")

        error = th.sum((grad_nn - grad_nl)**2).item()
        assert error < 1e-6

    def test_batched_iunfold(self):

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
        stride = args.stride

        # -- batching info --
        device = noisy.device
        shape = noisy.shape
        t,c,h,w = shape
        npix = t * h * w
        nh = (h-1)//stride+1
        nw = (w-1)//stride+1
        qTotal = t * nh * nw
        qSize = 128
        nbatches = (qTotal-1) // qSize + 1
        vid = vid.contiguous()

        # -- functions --
        # scatter_nl = dnls.scatter.ScatterNl(ps,pt,dilation=dil,exact=True)
        # coords = (2,2,h-2,w-2)
        coords = (12,12,h-12,w-12)
        coords = [2,2,h-6,w-12]
        iunfold_nl = dnls.iunfold.iUnfold(ps,coords,stride=stride,dilation=dil)

        # -- get check bounds --
        psHalf = ps//2
        a,b,c,d = coords # top,left,btm,right
        a = (a+psHalf-1)//stride+1
        b = (b+psHalf-1)//stride+1
        c = (c-psHalf-1)//stride+1
        d = (d-psHalf-1)//stride+1

        # -- prepare videos --
        psHalf = ps//2
        padf = psHalf*dil
        # vid = pad(vid,4*[padf,],mode="reflect")
        vid_nn = vid.clone()
        vid_nl = vid.clone()
        vid_nn = vid_nn.requires_grad_(True)
        vid_nl = vid_nl.requires_grad_(True)

        # -- prepare video with boarder --
        top,left,btm,right = coords
        hp = coords[2] - coords[0]
        wp = coords[3] - coords[1]
        padf = (h-hp)//2
        padp = dil*(ps//2)
        pad_tot = padf + padp
        vid_nn_cc = vid_nn[:,:,top:btm,left:right].contiguous()
        vid_nn_mod = pad(vid_nn_cc,[padp,]*4,mode="reflect")
        padfs = [left,w-right,top,h-btm]
        vid_nn_mod = pad(vid_nn_mod,padfs,mode="constant",value=0.)

        # -- exec fold fxns --
        # vid_nl = vid.clone().requires_grad_(True)
        patches_nl = []
        for index in range(nbatches):

            # -- get [patches & nlInds] --
            qindex = min(qSize * index,qTotal)
            qSize = min(qSize,qTotal-qindex)
            queryInds = dnls.utils.inds.get_query_batch(qindex,qSize,stride,
                                                        t,h,w,device)
            # nlDists,nlInds = dnls.simple.search.run(vid,queryInds,
            #                                         flow,k,ps,pt,ws,wt,chnls)

            # -- run forward --
            th.cuda.synchronize()
            patches_nl_i = iunfold_nl(vid_nl,qindex,qSize)
            th.cuda.synchronize()

            # -- agg for testing --
            patches_nl.append(patches_nl_i)

        # -- cat for testing --
        patches_nl = th.cat(patches_nl,0)

        # -- run fold with entire image --
        patches_nn = self.run_unfold(vid_nn_mod,ps,stride,dil)

        # -- save ex --
        viz_nn,w_nn = self.run_fold(patches_nn,t,h,w,stride,dil)
        # ones = self.sq_ones(patches_nl,t,h,w,coords,stride)
        viz_nl,w_nl = self.run_fold(patches_nl,t,h,w,stride,dil)#,ones)
        viz_nn /= w_nn
        viz_nl /= w_nl
        # print(viz_nn[0,0,28:32,28:32])
        # print(viz_nl[0,0,28:32,28:32])
        # print(th.where(patches_nl>0))
        # print(th.where(viz_nl>0))
        # vid_nn_s = viz_nn / viz_nn.max()
        # vid_nl_s = viz_nl / viz_nl.max()
        dnls.testing.data.save_burst(viz_nn,SAVE_DIR,"vid_nn")
        dnls.testing.data.save_burst(viz_nl,SAVE_DIR,"vid_nl")
        # psHalf = ps//2
        # diff = th.abs(vid_nn_s - vid_nl_s)
        # diff /= diff.max()
        # dnls.testing.data.save_burst(diff,SAVE_DIR,"diff")

        # -- run backward --
        # t,c,h,w = noisy.shape
        shape_str = '(t h w) 1 1 c ph pw -> t h w c ph pw'
        patches_nn = rearrange(patches_nn,shape_str,t=t,h=nh)
        patches_nl = rearrange(patches_nl,shape_str,t=t,h=nh)
        patches_grad = th.rand_like(patches_nn).type(th.float32)
        th.autograd.backward(patches_nn[:,a:c,b:d],patches_grad[:,a:c,b:d])
        th.autograd.backward(patches_nl[:,a:c,b:d],patches_grad[:,a:c,b:d])

        # -- vis --
        # print("\n")
        # print(patches[0,0,0,0])
        # print(patches[1,0,0,0])
        # print("-"*20)
        # print(vid_nn[0,0,:3,:3])
        # print("-"*20)
        # print(vid_nl[0,0,:3,:3])

        # -- check forward --
        error = th.sum((patches_nn[:,a:c,b:d] - patches_nl[:,a:c,b:d])**2).item()
        assert error < 1e-10

        # -- unpack grads --
        hp = coords[2] - coords[0]
        wp = coords[3] - coords[1]
        grad_nn = center_crop(vid_nn.grad,(hp,wp))
        grad_nl = center_crop(vid_nl.grad,(hp,wp))

        # -- view errors --
        # print(grad_nn[0,0,:7,:7])
        # print(grad_nl[0,0,:7,:7])

        # -- check backward --
        error = th.sum((grad_nn - grad_nl)**2).item()
        assert error < 1e-6

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
        args = edict({"ps":5,"pt":1,"k":1,"ws":10,"wt":5,
                      "stride":3,"dilation":1})
        return dname,sigma,comp_flow,args

    def sq_ones(self,patches,t,h,w,coords,stride):
        shape_str = '(t h w) 1 1 c ph pw -> t h w c ph pw'
        hs,ws = h//stride,w//stride
        patches = rearrange(patches,shape_str,h=hs,w=ws)
        top,left,btm,right = coords
        ones = th.zeros_like(patches)
        ones[:,top-1:btm,left-1:right,:] = 1
        ones = rearrange(ones,'t h w c ph pw -> t (c ph pw) (h w)')
        return ones

    def run_fold(self,patches,t,h,w,stride=1,dil=1,ones=None):
        ps = patches.shape[-1]
        psHalf = ps//2
        padf = dil * psHalf
        hp,wp = h+2*padf,w+2*padf
        shape_str = '(t np) 1 1 c h w -> t (c h w) np'
        patches = rearrange(patches,shape_str,t=t)
        if ones is None:
            ones = th.ones_like(patches)
        vid_pad = fold(patches,(hp,wp),(ps,ps),stride=stride,dilation=dil)
        vid = center_crop(vid_pad,(h,w))
        wvid_pad = fold(ones,(hp,wp),(ps,ps),stride=stride,dilation=dil)
        wvid = center_crop(wvid_pad,(h,w))

        return vid,wvid

    def run_unfold(self,vid_pad,ps,stride=1,dil=1):
        # psHalf = ps//2
        # vid_pad = pad(vid,4*[psHalf,],mode="reflect")
        shape_str = 't (c h w) np -> (t np) 1 1 c h w'
        patches = unfold(vid_pad,(ps,ps),stride=stride,dilation=dil)
        patches = rearrange(patches,shape_str,h=ps,w=ps)
        return patches

