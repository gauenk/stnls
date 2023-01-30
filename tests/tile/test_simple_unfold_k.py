
# -- python --
import sys

# -- data mgnmt --
from pathlib import Path
from easydict import EasyDict as edict

# -- testing --


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
# -- Primary Testing Loop --
#

def exec_patch_strided_test(dname,sigma,flow_args,args):
    """
    Check equality for patch-strided unfold_k
    """

    # -- misc --
    viz_mask = True

    # -- load data --
    device = args.device
    clean = dnls.testing.data.load_burst("./data",dname)[:10]
    clean = th.from_numpy(clean).to(device)
    noisy = clean + sigma * th.randn_like(clean)
    flow = dnls.flow.get_flow(flow_args.comp_flow,flow_args.clean_flow,
                                      noisy,clean,sigma)

    # -- unpack params --
    k = args.k
    ps = args.ps
    pt = args.pt
    ws = args.ws
    wt = args.wt
    chnls = args.chnls
    stride = args.stride

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

    # -- vizualize query inds --
    if VIZ and viz_mask:
        mask = th.zeros((t,c,h,w),device=device)
        mask[queryInds[:,0],:,queryInds[:,1],queryInds[:,2]] = 1
        dnls.testing.data.save_burst(mask,SAVE_DIR,"mask")

    # -- nl search --
    nlDists,nlInds = dnls.simple.search.run(clean,queryInds,
                                            flow,k,ps,pt,ws,wt,chnls,
                                            stride=1)#args.stride)
    patches = dnls.simple.unfold_k.run(clean,nlInds,ps,pt)
    patches = patches[:,[0]]
    nlDists = nlDists[:,[0]]
    nlInds = nlInds[:,[0]]
    # print(patches.shape)
    # patches = rearrange(patches[:,0,0],'(t q) c h w -> t (c h w) q',t=t)

    # -- get patches with unfold --
    pad,dil = ps//2,(args.stride,args.stride)
    clean_pad = pad_fxn(clean,(pad,pad,pad,pad),padding_mode="reflect")
    patches_uf = unfold(clean_pad,(ps,ps),dilation=dil)

    # -- fold with k = 1 --
    hp,wp = h+2*pad,w+2*pad
    ones = th.ones_like(patches_uf)
    Z = fold(ones,(hp,wp),(ps,ps),dilation=dil)
    vid,wvid = dnls.simple.fold_k.run(patches,nlDists,nlInds,shape=shape)
    vid_ss = vid / wvid
    vid_uf = fold(patches_uf,(hp,wp),(ps,ps),dilation=dil) / Z

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
    error = th.mean((vid_uf - clean)**2).item()
    assert error < 1e-10
    error = th.mean((vid_ss - clean)**2).item()
    assert error < 1e-10
    error = th.mean((vid_ss - vid_uf)**2).item()
    assert error < 1e-10
    error = th.max((vid_ss - vid_uf)**2).item()
    assert error < 1e-10

def exec_query_strided_test(dname,sigma,flow_args,args):
    """
    Check equality for strided unfold_k
    """

    # -- misc --
    viz_mask = True

    # -- load data --
    device = args.device
    clean = dnls.testing.data.load_burst("./data",dname)[:10]
    clean = th.from_numpy(clean).to(device)
    noisy = clean + sigma * th.randn_like(clean)
    flow = dnls.flow.get_flow(flow_args.comp_flow,flow_args.clean_flow,
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
    qStride = 5
    qSearchTotal_t = npix_t // qStride # _not_ a DivUp
    qSearchTotal = t * qSearchTotal_t
    qSearch = qSearchTotal
    nbatches = (qSearchTotal - 1) // qSearch + 1

    # -- get patches with search --
    index = 0
    queryInds = dnls.utils.inds.get_query_batch(index,qSearch,qStride,t,h,w,device)

    # -- vizualize query inds --
    if VIZ and viz_mask:
        mask = th.zeros((t,c,h,w),device=device)
        mask[queryInds[:,0],:,queryInds[:,1],queryInds[:,2]] = 1
        dnls.testing.data.save_burst(mask,SAVE_DIR,"mask")

    # -- nl search --
    nlDists,nlInds = dnls.simple.search.run(clean,queryInds,
                                            flow,k,ps,pt,ws,wt,chnls)
    patches = dnls.simple.unfold_k.run(clean,nlInds,ps,pt)
    patches = patches[:,[0]]
    nlDists = nlDists[:,[0]]
    nlInds = nlInds[:,[0]]
    # print(patches.shape)
    # patches = rearrange(patches[:,0,0],'(t q) c h w -> t (c h w) q',t=t)

    # -- get patches with unfold --
    pad = ps//2
    clean_pad = pad_fxn(clean,(pad,pad,pad,pad),padding_mode="reflect")
    patches_uf = unfold(clean_pad,(ps,ps))

    # -- fold with k = 1 --
    hp,wp = h+2*pad,w+2*pad
    ones = th.ones_like(patches_uf)
    Z = fold(ones,(hp,wp),(ps,ps))
    vid,wvid = dnls.simple.fold_k.run(patches,nlDists,nlInds,shape=shape)
    vid_ss = vid / wvid
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
    error = th.mean((vid_uf - clean)**2).item()
    assert error < 1e-10
    error = th.mean((vid_ss - clean)**2).item()
    assert error < 1e-10
    error = th.mean((vid_ss - vid_uf)**2).item()
    assert error < 1e-10
    error = th.max((vid_ss - vid_uf)**2).item()
    assert error < 1e-10

def exec_folding_test(dname,sigma,flow_args,args):
    """
    Check that "fold" === "fold_k" in exh search?
    """

    # -- load data --
    device = args.device
    clean = dnls.testing.data.load_burst("./data",dname)[:10]
    clean = clean[:,:,:32,:32]
    clean = th.from_numpy(clean).to(device)
    noisy = clean + sigma * th.randn_like(clean)
    flow = dnls.flow.get_flow(flow_args.comp_flow,flow_args.clean_flow,
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
    patches = dnls.simple.unfold_k.run(clean,nlInds,ps,pt)
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

#
# -- Launcher --
#

def test_simple_search():

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
    args = edict({'ps':7,'pt':1,'k':3,'ws':10,'stride':1,
                  'wt':5,'chnls':3,'device':device})
    flow_args = edict({'comp_flow':False,'clean_flow':False})
    # exec_folding_test(dname,sigma,flow_args,args)
    # exec_query_strided_test(dname,sigma,flow_args,args)

    args.stride = 2
    # exec_patch_strided_test(dname,sigma,flow_args,args)
    th.cuda.synchronize()
