#!/home/gauenk/.pyenv/shims/python
"""

Profile xsearch

"""

# -- python --
import sys

# -- data mgnmt --
from pathlib import Path
from easydict import EasyDict as edict

# -- testing --
import unittest,pytest

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- dnls --
import dnls
import dnls.utils.gpu_mem as gpu_mem
from dnls.utils.pads import comp_pads

# -- meshgrid --
import cache_io

# -- profiling --
import nvtx
from torch.profiler import profiler,record_function,ProfilerActivity

# -- test func --
# import torch.autograd.profiler as profiler,record_function,ProfilerActivity
from torch.nn.functional import fold,unfold,pad
from torchvision.transforms.functional import center_crop

# -- paths --
SAVE_DIR = Path("./output/tests/")

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)

def run_xsearch(ps,stride,dilation,**kwargs):

    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    chnls,k,pt = 1,1,1
    wt = 0
    ws = -1
    k = -1
    stride0 = stride
    stride1 = 1

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    exact = True
    exact = True
    gpu_stats = False
    adj = False

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device)[:1,].contiguous()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    vid = th.cat([vid,vid],-1)
    vid = th.cat([vid,vid],-1)
    vid = th.cat([vid,vid],-2)
    vid = th.cat([vid,vid],-2)
    print("vid.shape: ",vid.shape)

    # -- normalize --
    vid /= vid.max()

    # -- compute flow --
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape
    chnls = vid.shape[1]

    # -- sub square --
    top,btm,left,right = 0,h,0,w
    coords = [top,left,btm,right]
    # sq_h = coords[2] - coords[0]
    # sq_w = coords[3] - coords[1]

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid.shape, ps, stride0, dil)
    oh1,ow1,_,_ = comp_pads(vid.shape, ps, stride1, dil)
    n_h = (hp - (ps-1)*dil - 1)//stride0 + 1
    n_w = (wp - (ps-1)*dil - 1)//stride0 + 1

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h * n_w
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- swap --
    oh0,ow0,_,_ = comp_pads(vid.shape, ps, stride0, dil)
    oh1,ow1,_,_ = comp_pads(vid.shape, ps, stride1, dil)

    # -- exec fold fxns --
    xsearch = dnls.xsearch.CrossSearchNl(flows.fflow, flows.bflow, k, ps, pt,
                                         ws, wt, oh0, ow0, oh1, ow1,
                                         chnls=chnls,dilation=dil, stride=stride1,
                                         use_bound=True,use_k=False,exact=True)
    # -- query inds
    qindex = 0
    iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch,stride0,
                                                coords,t,device)
    # -- binary image to remove float error --
    # vidr = None
    # vidr = 10*th.ones_like(vid)
    # vid = th.round(th.rand_like(vid),decimals=2)*100
    # vid = th.round(th.rand_like(vid),decimals=3)
    vidr = th.round(th.rand_like(vid),decimals=3)
    # vid = th.round(th.rand_like(vid),decimals=2)*100.
    # vidr = th.round(th.rand_like(vid),decimals=2)*100.
    # vid = vid.type(th.float32)
    # vidr = vidr.type(th.float32)
    # vidr[th.where(th.abs(vidr) > 0.2)] = 1
    # vidr[th.where(th.abs(vidr) < 1)] = 0
    # # vid = th.ones_like(vid)
    # vid = th.rand_like(vid)
    # vid[th.where(th.abs(vid) > 0.2)] = 1
    # vid[th.where(th.abs(vid) < 1)] = 0

    # vidr[...] = 1
    # vid = vidr.clone()
    # vid[:,:,:3,:3] = 0
    # vid[:,:,0,0] = 0
    # vidr[:,:,:3,:3] = 0


    # -- allow grads --
    vid_cu = vid.clone()
    vidr_cu = vidr.clone()
    vid_nn = vid.clone()
    vidr_nn = vidr.clone()
    vid_cu.requires_grad_(True)
    vidr_cu.requires_grad_(True)
    vid_nn.requires_grad_(True)
    vidr_nn.requires_grad_(True)

    #
    # -- run search --
    #

    # -- run cu --
    _,_ = xsearch(vid_cu,iqueries,vid1=vidr_cu) # warm-up
    with nvtx.annotate("xsearch",color="purple"):
        nlDists_cu,nlInds_cu = xsearch(vid_cu,iqueries,vid1=vidr_cu)
    # activs = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    # with profiler.profile(activities=activs,record_shapes=True,with_stack=True, profile_memory=True) as prof:
    #     with nvtx.annotate("xsearch",color="purple"):
    #         nlDists_cu,nlInds_cu = xsearch(vid_cu,iqueries,vid1=vidr_cu)
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
    nlDists_cu = rearrange(nlDists_cu,'(sh sw) (h w) -> h w sh sw',sh=n_h,h=h)
    # nlDists_cu = rearrange(nlDists_cu,'(sh sw) h w -> h w sh sw',sh=n_h)
    exit(0)

    # -- run nn --
    nlDists_nn,nlInds_nn = dnls.simple.xsearch_nn.run_nn(vid_nn,ps,stride=stride0,
                                                         dilation=dil,vid1=vidr_nn)
    sh = nlDists_nn.shape[-1]

    # -- vis --
    diff = th.abs(nlDists_cu - nlDists_nn)
    args = th.where(diff>1e-10)
    for i in range(len(args)):
        print(i,th.unique(args[i]))
    if diff.max() > 1e-10: diff /= diff.max()
    dnls.testing.data.save_burst(diff[0,0][None,None],"./output/tests/xsearch/","diff")
    dnls.testing.data.save_burst(diff[:,:,0,0][None,None],"./output/tests/xsearch/","diff_d00")

    # -- compare fwd --
    max_error = th.abs(nlDists_cu - nlDists_nn).max().item()
    print("max error: ",max_error)
    assert max_error < 1e-3

    error = th.mean(th.abs(nlDists_cu - nlDists_nn)).item()
    print("error: ",error)
    assert error < 1e-4

    # -- compute grad --
    nlDists_grad = th.ones_like(nlDists_nn)
    th.autograd.backward(nlDists_nn,nlDists_grad)
    th.autograd.backward(nlDists_cu,nlDists_grad)

    # -- get grads --
    grads_cu = vidr_cu.grad
    grads_nn = vidr_nn.grad
    # print(grads_cu.shape)
    # print(grads_nn.shape)
    # print("cu,nn")
    # print("-"*10)
    # print(grads_cu[0,0,:3,:3])
    # print(grads_nn[0,0,:3,:3])
    # print("-"*10)
    # print(grads_cu[0,0,16:19,16:19])
    # print(grads_nn[0,0,16:19,16:19])
    # print("-"*10)
    # print(grads_cu[0,0,30:33,30:33])
    # print(grads_nn[0,0,30:33,30:33])
    # print("-"*10)
    # print(grads_cu[0,0,-3:,-3:])
    # print(grads_nn[0,0,-3:,-3:])
    # print("-"*10)

    # -- compare grads --
    rel_error = th.abs(grads_nn - grads_cu)/(th.abs(grads_nn)+1e-8)
    args = th.where(th.abs(grads_nn) > 1e-3)
    rel_error_nz = rel_error[args]
    args_z = th.where(th.abs(grads_nn) <= 1e-3)
    # print(args_z)

    # args = th.where(rel_error> 0.5)
    # print(th.sum(th.abs(grads_cu[args])))
    # print(th.sum(th.abs(grads_nn[args])))
    # print(grads_cu[args],grads_nn[args])
    # print(len(args[0]))
    # print(args)

    error = th.max(rel_error_nz).item()
    print("Max Error: ",error)
    assert error < 1e-4

    error = th.mean(rel_error_nz).item()
    print("Mean Error: ",error)
    assert error < 1e-4

    # error = th.max(th.abs(grads_cu[args_z])).item()
    # print("Max Error: ",error)
    # assert error < 1e-4

    # error = th.mean(th.abs(grads_cu[args_z])).item()
    # print("Mean Error: ",error)
    # assert error < 1e-4


    #
    # -- compare --
    #

    # -- get grads --
    grads_cu = vid_cu.grad
    grads_nn = vid_nn.grad

    # -- viz --
    # print(grads_cu.shape)
    # print(grads_nn.shape)
    # print("cu,nn")
    # print("-"*10)
    # print(grads_cu[0,0,:3,:3])
    # print(grads_nn[0,0,:3,:3])
    # print("-"*10)
    # print(grads_cu[0,0,16:19,16:19])
    # print(grads_nn[0,0,16:19,16:19])
    # print("-"*10)
    # print(grads_cu[0,0,30:33,30:33])
    # print(grads_nn[0,0,30:33,30:33])
    # print("-"*10)
    # print(grads_cu[0,0,-3:,-3:])
    # print(grads_nn[0,0,-3:,-3:])
    # print("-"*10)

    # -- compare grads --
    args = th.where(th.abs(grads_nn) > 1e-8)
    rel_error_nz = rel_error[args]
    args_z = th.where(th.abs(grads_nn) <= 1e-8)
    print(args_z)

    error = th.max(rel_error_nz).item()
    print("Max Error: ",error)
    assert error < 1e-4

    error = th.mean(rel_error_nz).item()
    print("Mean Error: ",error)
    assert error < 1e-4

    # error = th.max(th.abs(grads_cu[args_z])).item()
    # print("Max Error: ",error)
    # assert error < 1e-4

    # error = th.mean(th.abs(grads_cu[args_z])).item()
    # print("Mean Error: ",error)
    # assert error < 1e-4


def main():
    # -- seed --
    seed = 123
    set_seed(seed)

    # -- tests --
    lists = {"ps":[7],"stride":[4],"dilation":[1],"wt":[0],
             "ws":[-1],"top":[0],"btm":[64],"left":[0],"right":[64]}
    # lists = {"ps":[3,4,5,6,7,8],"stride":[1,2,3,4,5,8],"dilation":[1,2,3,4,5,8],
    #          "top":[1,11],"btm":[50,57],"left":[3,7],"right":[57,30]}

    # -- mesh --
    exps = cache_io.mesh_pydicts(lists) # create mesh


    # -- for exp in exps --
    for exp in exps:
        run_xsearch(**exp)

if __name__ == "__main__":
    main()
