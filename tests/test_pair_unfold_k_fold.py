
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

class TestPairUnfoldKFoldK(unittest.TestCase):

    #
    # -- Test v.s. NN --
    #

    def test_nn(self):

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
        qTotal = t * (h//stride) * (w//stride)
        qSize = qTotal
        nbatches = (qTotal-1) // qSize + 1
        vid = vid.contiguous()
        vid = th.randn_like(vid)

        # -- exec unfold fxns --
        unfold_k = dnls.UnfoldK(ps,pt,dilation=dil,exact=True)
        fold_nl = dnls.Fold((t,c,h,w),stride=stride,dilation=dil)

        # -- prepare videos --
        psHalf = ps//2
        padf = psHalf*dil
        vid_pad = pad(vid,4*[padf,],mode="reflect")
        vid_nn = vid_pad.clone()
        vid_nl = vid_pad.clone()
        vid_nn.requires_grad_(True)
        vid_nl.requires_grad_(True)
        vid_nl_cc = center_crop(vid_nl,(h,w)).contiguous()

        # -- weights --
        weight = th.rand((1,1,1,c,ps,ps),device=device,dtype=th.float32)
        weight_nn = weight.clone().requires_grad_(True)
        weight_nl = weight.clone().requires_grad_(True)
        # wpatches_nn = patches_nn
        # wpatches_nl = patches_nl


        #
        # -- test logic --
        #

        # -- compute search --
        index = 0
        queryInds = dnls.utils.inds.get_query_batch(index,qSize,stride,
                                                    t,h,w,device)
        nlDists,nlInds = dnls.simple.search.run(vid,queryInds,
                                                flow,k,ps,pt,ws,wt,chnls)

        # -- run forward --
        assert k == 1,"Must have k == 1 for test."
        patches_nn = self.run_unfold(vid_nn,ps,stride,dil)
        patches_nl = unfold_k(vid_nl_cc,nlInds[:,[0]])
        assert th.sum((patches_nn - patches_nl)**2).item() < 1e-10
        # patches_nl = unfold_nl(vid_nl_cc,0,qTotal) # k == 1 only

        # -- modify --
        wpatches_nn = weight_nn * patches_nn
        wpatches_nl = weight_nl * patches_nl

        # -- run forward --
        vid_nn_out,_ = self.run_fold(wpatches_nn,t,h,w,stride,dil)
        vid_nl_out = fold_nl(wpatches_nl,0)

        # -- run backward --
        vid_grad = th.rand_like(vid_nn_out).type(th.float32)
        th.autograd.backward(vid_nn_out,vid_grad)
        th.autograd.backward(vid_nl_out,vid_grad)

        # -- check forward --
        error = th.sum((vid_nn_out - vid_nl_out)**2).item()/ps
        assert error < 1e-6

        # -- check backward --
        grad_nn = center_crop(vid_nn.grad,(h,w))
        grad_nl = center_crop(vid_nl.grad,(h,w))

        # -- compute error --
        diff = th.abs(grad_nn - grad_nl)
        dmax = diff.max()
        if dmax > 1e-3: diff /= dmax
        dnls.testing.data.save_burst(diff,SAVE_DIR,"diff")

        error = th.sum((grad_nn - grad_nl)**2).item()/ps
        assert error < 1e-6
        th.cuda.synchronize()

    def test_batched(self):

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
        qTotal = t * (h//stride) * (w//stride)
        qSize = qTotal
        nbatches = (qTotal-1) // qSize + 1
        vid = vid.contiguous()
        vid = th.randn_like(vid)

        # -- exec unfold fxns --
        unfold_k = dnls.UnfoldK(ps,pt,dilation=dil,exact=True)
        fold_nl = dnls.Fold((t,c,h,w),stride=stride,dilation=dil)

        # -- prepare videos --
        psHalf = ps//2
        padf = psHalf*dil
        vid_pad = pad(vid,4*[padf,],mode="reflect")
        vid_nn = vid_pad.clone()
        vid_nl = vid_pad.clone()
        vid_nn.requires_grad_(True)
        vid_nl.requires_grad_(True)
        vid_nl_cc = center_crop(vid_nl,(h,w)).contiguous()

        # -- init weights --
        weight = th.rand((1,1,1,c,ps,ps),device=device,dtype=th.float32)
        weight_nn = weight.clone().requires_grad_(True)
        weight_nl = weight.clone().requires_grad_(True)

        # -- exec fold fxns --
        for index in range(nbatches):

            # -- compute search --
            qindex = min(index * qSize,qTotal)
            qSize = min(qSize,qTotal - qindex)
            queryInds = dnls.utils.inds.get_query_batch(qindex,qSize,stride,
                                                        t,h,w,device)
            nlDists,nlInds = dnls.simple.search.run(vid,queryInds,
                                                    flow,k,ps,pt,ws,wt,chnls)

            # -- run forward --
            assert k == 1,"Must have k == 1 for test."
            patches_nl_i = unfold_k(vid_nl_cc,nlInds[:,[0]])

            # -- modifiy --
            wpatches_nl_i = weight_nl * patches_nl_i

            # -- run forward --
            vid_nl_out = fold_nl(wpatches_nl_i,0)

        # -- cat for testing --
        vid_nl_out = fold_nl.vid

        # -- run fold with entire image --
        patches_nn = self.run_unfold(vid_nn,ps,stride,dil)
        wpatches_nn = weight_nn * patches_nn
        vid_nn_out,_ = self.run_fold(wpatches_nn,t,h,w,stride,dil)

        # -- run backward --
        vid_grad = th.rand_like(vid_nn_out).type(th.float32)
        th.autograd.backward(vid_nn_out,vid_grad)
        th.autograd.backward(vid_nl_out,vid_grad)

        # -- check forward --
        error = th.sum((vid_nn_out - vid_nl_out)**2).item()/ps
        assert error < 1e-6

        # -- check backward --
        grad_nn = center_crop(vid_nn.grad,(h,w))
        grad_nl = center_crop(vid_nl.grad,(h,w))

        # -- compute error --
        diff = th.abs(grad_nn - grad_nl)
        dmax = diff.max()
        if dmax > 1e-3: diff /= dmax
        dnls.testing.data.save_burst(diff,SAVE_DIR,"diff")

        error = th.sum((grad_nn - grad_nl)**2).item()/ps
        assert error < 1e-6
        th.cuda.synchronize()

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
        args = edict({"ps":11,"pt":1,"k":1,"ws":10,"wt":5,
                      "stride":2,"dilation":2})
        return dname,sigma,comp_flow,args

    def run_fold(self,patches,t,h,w,stride=1,dil=1):
        ps = patches.shape[-1]
        psHalf = ps//2
        padf = dil * psHalf
        hp,wp = h+2*padf,w+2*padf
        shape_str = '(t np) 1 1 c h w -> t (c h w) np'
        patches = rearrange(patches,shape_str,t=t)
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

