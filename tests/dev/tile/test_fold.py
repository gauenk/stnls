
# -- python --
import sys

# -- data mgnmt --
from pathlib import Path
from easydict import EasyDict as edict

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- stnls --
import stnls

# -- test func --
from torch.nn.functional import fold,unfold,pad
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/")

#
# -- Primary Testing Class --
#

def test_nn_fold():

    # -- get args --
    dname,ext,sigma,comp_flow,args = setup()

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    exact = True

    # -- load data --
    vid = stnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device)
    noisy = vid + sigma * th.randn_like(vid)
    flow = stnls.flow.get_flow(comp_flow,clean_flow,noisy,vid,sigma)

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
    nh = int((h-1) // stride) + 1
    nw = int((w-1) // stride) + 1
    qSize = t * nh * nw
    qTotal = t * nh * nw
    nbatches = (qTotal-1) // qSize + 1
    vid = vid.contiguous()

    # -- exec fold fxns --
    unfold_k = stnls.UnfoldK(ps,pt,dilation=dil,exact=True)
    fold_nl = stnls.Fold((t,c,h,w),stride=stride,dilation=dil)

    # -- get [patches & nlInds] --
    index = 0
    queryInds = stnls.utils.inds.get_query_batch(index,qSize,stride,
                                                t,h,w,device)
    nlDists,nlInds = stnls.simple.search.run(vid,queryInds,
                                            flow,k,ps,pt,ws,wt,chnls)
    patches = unfold_k(vid,nlInds)
    # patches_uf = run_unfold(vid,ps,stride=stride,dil=dil)
    # assert th.sum((patches-patches_uf)**2).item() < 1e-10
    # th.cuda.synchronize()

    # -- save query mask --
    mask = th.zeros((t,h,w),dtype=np.bool,device=device)
    start = h*w
    mask[queryInds[:,0],queryInds[:,1],queryInds[:,2]] = 1
    mask = repeat(mask,'t h w -> t c h w',c=3)
    stnls.testing.data.save_burst(mask,SAVE_DIR,"mask")
    assert th.sum(queryInds - nlInds[:,0]) < 1e-10

    #
    # -- test logic --
    #

    # -- prepare videos --
    patches_nn = patches
    patches_nl = patches.clone()
    patches_nn.requires_grad_(True)
    patches_nl.requires_grad_(True)

    # -- run forward --
    vid_nn,_ = run_fold(patches_nn,t,h,w,stride,dil)
    vid_nl = fold_nl(patches_nl,0)

    # -- run backward --
    vid_grad = th.randn_like(vid)
    th.autograd.backward(vid_nn,vid_grad)
    th.autograd.backward(vid_nl,vid_grad)

    # -- save ex --
    vid_nn_s = vid_nn / vid_nn.max()
    vid_nl_s = vid_nl / vid_nl.max()
    # print("vid_nl.max().item(): ",vid_nl.max().item())
    # print("vid_nn.max().item(): ",vid_nn.max().item())
    stnls.testing.data.save_burst(vid_nn_s,SAVE_DIR,"vid_nn")
    stnls.testing.data.save_burst(vid_nl_s,SAVE_DIR,"vid_nl")
    diff = th.abs(vid_nn_s - vid_nl_s)
    diff /= diff.max()
    stnls.testing.data.save_burst(diff,SAVE_DIR,"diff")

    # -- vis --
    # print("\n")
    # print(patches[0,0,0,0])
    # print(patches[1,0,0,0])
    # print("-"*20)
    # print(vid_nn[0,0,:3,:3])
    # print("-"*20)
    # print(vid_nl[0,0,:3,:3])

    # -- check forward --
    error = th.sum((vid_nn - vid_nl)**2).item()
    # hm,wm = h-ps,w-ps
    # error = th.mean((center_crop(vid_nn - vid_nl,(hm,wm)))**2).item()
    assert error < 1e-10

    # -- check backward --
    grad_nn = patches_nn.grad
    grad_nl = patches_nl.grad

    # -- inspect grads --
    # print("grad_nn.shape: ",grad_nn.shape)
    # print(grad_nn[0,0,0,0])
    # print(grad_nl[0,0,0,0])
    # print("-"*10)
    # print(grad_nn[1,0,0,0])
    # print(grad_nl[1,0,0,0])
    # print("-"*10)
    # print(grad_nn[100,0,0,0])
    # print(grad_nl[100,0,0,0])
    # print("-"*10)
    # print(grad_nn[200,0,0,0])
    # print(grad_nl[200,0,0,0])

    # -- check backward --
    error = th.sum((grad_nn - grad_nl)**2).item()
    assert error < 1e-10
    # print("GPU Max: ",th.cuda.memory_allocated()/(1024**3))

#
# -- Test v.s. NN --
#

# @pytest.mark.parametrize("a,b,expected", testdata)
def test_batched_fold():

    # -- get args --
    dname,ext,sigma,comp_flow,args = setup()

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    exact = True
    gpu_stats = False

    # -- load data --
    vid = stnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device)
    noisy = vid + sigma * th.randn_like(vid)
    flow = stnls.flow.get_flow(comp_flow,clean_flow,noisy,vid,sigma)

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
    nh = int((h-1) // stride) + 1
    nw = int((w-1) // stride) + 1
    qTotal = t * nh * nw
    qSize = qTotal//2
    nbatches = (qTotal-1) // qSize + 1
    vid = vid.contiguous()
    if gpu_stats:
        print("nbatches: ",nbatches)

    # -- vis --
    if gpu_stats:
        gpu_max = th.cuda.memory_allocated()/(1024**3)
        print("[pre-def] GPU Max: %2.4f" % (gpu_max))

    # -- exec fold fxns --
    unfold_k = stnls.UnfoldK(ps,pt,dilation=dil,exact=True)
    fold_nl = stnls.Fold((t,c,h,w),stride=stride,dilation=dil)
    agg_patches = []
    # vid_nl = th.zeros((t,c,h,w),device=device)

    # -- vis --
    if gpu_stats:
        gpu_max = th.cuda.memory_allocated()/(1024**3)
        print("[pre-loop] GPU Max: %2.4f" % (gpu_max))

    for index in range(nbatches):

        # -- get [patches & nlInds] --
        qindex = min(qSize * index,qTotal)
        qSize = min(qSize,qTotal - qindex)
        queryInds = stnls.utils.inds.get_query_batch(qindex,qSize,stride,
                                                    t,h,w,device)
        nlDists,nlInds = stnls.simple.search.run(vid,queryInds,
                                                flow,k,ps,pt,ws,wt,chnls)
        patches_nl = unfold_k(vid,nlInds)
        del queryInds,nlDists,nlInds
        th.cuda.empty_cache()

        # -- prepare videos --
        patches_nl.requires_grad_(True)

        # -- print infos --
        # print("index: ",index)
        # print("qindex: ",qindex)
        # print("patches.shape: ",patches.shape)
        # wi = qindex % w
        # hi = (qindex // w) % h
        # ti = qindex // (h*w)
        # print("ti,hi,wi: ",ti,hi,wi)

        # -- run forward --
        # th.cuda.synchronize()
        vid_nl = fold_nl(patches_nl,qindex)
        # vid_nl_i = vid_nl
        # th.cuda.synchronize()
        # vid_nl += vid_nl_i

        # -- save --
        # vid_nl_p = vid_nl / (ps*ps)
        # stnls.testing.data.save_burst(vid_nl_p,SAVE_DIR,"vid_nl_%d" % index)

        # -- agg for testing --
        agg_patches.append(patches_nl)

        # -- vis --
        # if gpu_stats:
        #     th.cuda.synchronize()
        #     gpu_max = th.cuda.memory_allocated()/(1024**3)
        #     print("[%d]GPU Max: %2.4f" % (index,gpu_max))

    # -- cat for testing --
    # agg_patches = th.cat(agg_patches,0)

    # -- vis --
    if gpu_stats:
        th.cuda.synchronize()
        th.cuda.empty_cache()
        gpu_max = th.cuda.memory_allocated()/(1024**3)
        print("[post-loop] GPU Max: %2.4f" % (gpu_max))

    # -- run fold with entire image --
    index = 0
    nh = int((h-1) // stride) + 1
    nw = int((w-1) // stride) + 1
    qSize = t * (nh) * (nh)
    # qSize = t * (h//stride) * (w//stride)
    queryInds = stnls.utils.inds.get_query_batch(index,qSize,stride,
                                                t,h,w,device)
    nlDists,nlInds = stnls.simple.search.run(vid,queryInds,
                                            flow,k,ps,pt,ws,wt,chnls)
    # -- vis --
    del queryInds,nlDists
    if gpu_stats:
        th.cuda.synchronize()
        th.cuda.empty_cache()
        gpu_max = th.cuda.memory_allocated()/(1024**3)
        print("[post-search] GPU Max: %2.4f" % (gpu_max))

    # -- unfold_k --
    patches = unfold_k(vid,nlInds)
    th.cuda.synchronize()

    # -- vis --
    if gpu_stats:
        th.cuda.empty_cache()
        gpu_max = th.cuda.memory_allocated()/(1024**3)
        print("[post-unfold_k] GPU Max: %2.4f" % (gpu_max))

    # -- run nn --
    patches_nn = patches.clone()
    patches_nn.requires_grad_(True)
    vid_nn,_ = run_fold(patches_nn,t,h,w,stride,dil)

    # -- vis --
    if gpu_stats:
        gpu_max = th.cuda.memory_allocated()/(1024**3)
        print("[pre-bkwd] GPU Max: %2.4f" % (gpu_max))

    # -- run backward --
    # vid_nl = fold_nl.vid
    vid_grad = th.randn_like(vid)
    th.autograd.backward(vid_nn,vid_grad)
    th.autograd.backward(vid_nl,vid_grad)

    # -- vis --
    if gpu_stats:
        gpu_max = th.cuda.memory_allocated()/(1024**3)
        print("[post-bkwd] GPU Max: %2.4f" % (gpu_max))

    # -- save ex --
    # print("vid_nn.max(): ",vid_nn.max())
    # print("vid_nl.max(): ",vid_nl.max())
    vid_nn_s = vid_nn / vid_nn.max()
    vid_nl_s = vid_nl / vid_nl.max()
    stnls.testing.data.save_burst(vid_nn_s,SAVE_DIR,"vid_nn")
    stnls.testing.data.save_burst(vid_nl_s,SAVE_DIR,"vid_nl")
    psHalf = ps//2
    diff = th.abs(vid_nn_s - vid_nl_s)
    diff /= diff.max()
    stnls.testing.data.save_burst(diff,SAVE_DIR,"diff")

    # -- vis --
    # print("\n")
    # print(patches[0,0,0,0])
    # print(patches[1,0,0,0])
    # print("-"*20)
    # print(vid_nn[0,0,:3,:3])
    # print("-"*20)
    # print(vid_nl[0,0,:3,:3])

    # -- check forward --
    error = th.sum((vid_nn - vid_nl)**2).item()
    # hm,wm = h-2*dil*psHalf,w-2*dil*psHalf
    # error = th.mean((center_crop(vid_nn - vid_nl,(hm,wm)))**2).item()
    assert error < 1e-10

    # -- check backward --
    grad_nn = patches_nn.grad
    # grad_nl = patches_nl.grad
    # grad_nl = agg_patches.grad
    grad_nl = th.cat([p_nl.grad for p_nl in agg_patches])

    # -- inspect grads --
    # print("grad_nn.shape: ",grad_nn.shape)
    # print("grad_nl.shape: ",grad_nl.shape)
    # print(grad_nn[0,0,0,0])
    # print(grad_nl[0,0,0,0])
    # print(grad_nn[100,0,0,0])
    # print(grad_nl[100,0,0,0])
    # print(grad_nn[-1,0,0,0])
    # print(grad_nl[-1,0,0,0])
    # print(grad_nn[-3,0,0,0])
    # print(grad_nl[-3,0,0,0])

    # -- reshape --
    shape_str = '(t h w) 1 1 c ph pw -> t c h w ( ph pw)'
    grad_nn = rearrange(grad_nn,shape_str,t=t,h=h)
    grad_nl = rearrange(grad_nl,shape_str,t=t,h=h)
    # print("grad_nn.shape: ",grad_nn.shape)
    errors = th.mean((grad_nn - grad_nl)**2,dim=-1)
    # print("errors.shape: ",errors.shape)
    errors /= errors.max()
    # stnls.testing.data.save_burst(errors,SAVE_DIR,"errors")

    # -- view errors --
    # args = th.where(errors > 0)
    # print(grad_nn[args][:3])
    # print(grad_nl[args][:3])

    # -- check backward --
    error = th.sum((grad_nn - grad_nl)**2).item()
    assert error < 1e-10
    # print("GPU Max: ",th.cuda.max_memory_reserved()/(1024**3))

#
# -- Launcher --
#

def setup():

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
    dname,ext = "text_tourbus_64","jpg"
    dname,ext = "davis_baseball_64x64","jpg"
    # dname,ext = "text_bus","png"
    args = edict({"ps":5,"pt":1,"k":1,
                  "ws":10,"wt":5,
                  "stride":4,"dilation":2})
    return dname,ext,sigma,comp_flow,args

def run_fold(patches,t,h,w,stride=1,dil=1):
    th.cuda.synchronize()
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
    th.cuda.synchronize()

    return vid,wvid

def run_unfold(vid,ps,stride=1,dil=1):
    psHalf = ps//2
    padf = dil * psHalf
    shape_str = 't (c h w) np -> (t np) 1 1 c h w'
    vid_pad = pad(vid,4*[padf,],mode="reflect")
    patches = unfold(vid_pad,(ps,ps),stride=stride,dilation=dil)
    patches = rearrange(patches,shape_str,h=ps,w=ps)
    return patches

