"""

Create an error map of the backward step through colanet
between the exact and approximate gradient cuda kernels

"""


# -- misc --
import os,math,tqdm
import pprint,copy,random
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
import pandas as pd
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- optical flow --
import svnlb

# -- caching results --


# -- testing fxns --
import dnls
from dnls.utils.misc import rslice,assert_nonan
import dnls.utils.gpu_mem as gpu_mem
from dnls.utils.timer import ExpTimer
import torch.nn.functional as F


def run_experiment(cfg):

    # -- set seed --
    random.seed(cfg.seed)
    th.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda:0"

    # -- init log dir --
    log_dir = Path(cfg.log_root) / str(cfg.uuid)
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    # -- load data --
    data,loaders = data_hub.sets.load(cfg)
    sample = data.val[cfg.sample_index]
    index,region = sample['index'].item(),sample['region']
    noisy = rslice(sample['noisy'].to(cfg.device),region)/255.
    clean = rslice(sample['clean'].to(cfg.device),region)/255.
    print("noisy.shape: ",noisy.shape)

    # -- compute flow --
    comp_flow,clean_flow = cfg.flow == "true",False
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,noisy,noisy,0.)

    # -- mod channels --
    noisy = noisy[:,[0]].contiguous()
    noisy = repeat(noisy,'t 1 h w -> t c h w',c=cfg.nchnls)

    # -- batching info --
    stride0 = 4
    stride1 = 1
    vshape = noisy.shape
    t,c,h,w = vshape
    npix = t * h * w

    # -- get search size --
    region = [0,t,0,0,h,w]
    coords = region[2:]
    region = region[2:]
    cr_h = region[2] - region[0]
    cr_w = region[3] - region[1]
    dil,adj = 1,0

    # -- batching params --
    nh = (cr_h-1)//stride0+1
    nw = (cr_w-1)//stride0+1

    # -- pads --
    ps,pt = cfg.ps,1
    oh0,ow0 = 3,3#comp_pads(vshape, ps, stride0, dil)
    oh1,ow1,hp,wp = 1,1,(h+2*(ps//2)),(w+2*(ps//2))
    nh = (cr_h-1)//stride0+1
    nw = (cr_w-1)//stride0+1
    ntotal = t * nh * nw


    # -- run both --
    results = edict()
    for use_exact in [True,False]:

        # -- init search
        xsearch = dnls.xsearch.CrossSearchNl(flows.fflow, flows.bflow, cfg.k,
                                             cfg.ps, pt, cfg.ws, cfg.wt,
                                             oh0, ow0, oh1, ow1, chnls=cfg.nchnls,
                                             dilation=dil, stride=stride1,
                                             exact=use_exact)
        scatter = dnls.scatter.ScatterNl(ps,pt,dilation=dil,exact=use_exact)
        ifold = dnls.ifold.iFold(vshape,coords,stride=stride0,dilation=dil,adj=adj)
        wfold = dnls.ifold.iFold(vshape,coords,stride=stride0,dilation=dil,adj=adj)
        softmax_scale = 10.

        # -- forward --
        vid = noisy.clone()
        vid = vid.requires_grad_(True)

        # -- run batches --
        ntotal = int(t * nh * nw)
        nbatch = min(cfg.nbatch,ntotal) if cfg.nbatch > 0 else ntotal
        nbatches = (ntotal-1) // nbatch + 1
        print(nbatch,nbatches)

        # -- metrics --
        timer_agg = dnls.utils.timer.ExpTimer()
        timer_agg.start("forward")
        gpu_agg = gpu_mem.GpuRecord()
        gpu_agg.reset()

        # -- batch across queries --
        for index in range(nbatches):

            # -- timer --
            # print("%d/%d" % (index+1,nbatches))
            timer = dnls.utils.timer.ExpTimer()
            # gpu_rec = gpu_mem.GpuRecord()

            # -- batch info --
            qindex = min(nbatch * index,ntotal)
            nbatch_i =  min(nbatch, ntotal - qindex)

            # -- get patches --
            iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch_i,stride0,
                                                        region,t,device=device)
            th.cuda.synchronize()
            # print("iqueries.shape: ",iqueries.shape)

            # -- search --
            # gpu_rec.reset()
            timer.start("xsearch")
            nlDists_cu,nlInds_cu = xsearch(vid,iqueries,vid)
            timer.stop("xsearch")
            # gpu_rec.snap("xsearch")
            if nlDists_cu.ndim == 3:
                nlDists_cu = rearrange(nlDists_cu,'d0 h w -> d0 (h w)')

            # -- scatter --
            # gpu_rec.reset()
            yi = F.softmax(nlDists_cu*softmax_scale,1)
            yi = yi[...,None].type(th.float64)
            patches_i = scatter(vid,nlInds_cu).type(th.float64)
            # gpu_rec.snap("scatter")

            # -- compute weighted sum of top-k --
            # gpu_rec.reset()
            patches_i = rearrange(patches_i,'n k 1 c h w -> n k (c h w)')
            _,k,dim = patches_i.shape
            zi = th.sum(yi * patches_i,1).type(th.float32)
            assert_nonan(zi)
            # gpu_rec.snap("wsum")


            # -- ifold --
            timer.start("fold")
            # gpu_rec.reset()
            zi = rearrange(zi,'n (c h w) -> n 1 1 c h w',h=ps,w=ps)
            ones = th.ones_like(zi)
            ifold(zi,qindex)
            wfold(ones,qindex)
            # gpu_rec.snap("folds")
            timer.stop("fold")

            # -- vis --
            # print(gpu_rec)

        # -- normalize --
        y = ifold.vid
        Z = wfold.vid
        y = y/Z
        assert_nonan(y)
        timer_agg.stop("forward")
        gpu_agg.snap("forward")

        # -- backward --
        expected = th.ones_like(y)
        gpu_agg.reset()
        timer_agg.start("backward")
        th.autograd.backward(y,expected)
        timer_agg.stop("backward")
        gpu_agg.snap("backward")
        print(timer_agg)
        print(gpu_agg)

        # -- autograd collect --
        grad = vid.grad

        # -- modding string --
        estr = "exact" if use_exact else "not_exact"

        # -- save grad to dir --
        grad_root = Path("./output/bwd_error_map/grads/")
        file_stem = "%d_%d_%d.pt" % (cfg.sample_index,cfg.seed,cfg.rep_id)
        grad_dir = grad_root / estr
        if not grad_dir.exists(): grad_dir.mkdir(parents=True)
        grad_fn = str(grad_dir / file_stem)
        th.save(grad.cpu(),grad_fn)

        # -- results --
        results['%s_grad' % estr] = grad_fn
        # results['%s_loss' % estr] = loss.item()
        # results['%s_psnr' % estr] = psnr.item()
        results['%s_forward' % estr] = timer_agg['forward']
        results['%s_backward' % estr] = timer_agg['backward']
        results['%s_fwd_mem' % estr] = gpu_agg['forward']
        results['%s_bwd_mem' % estr] = gpu_agg['backward']


    results.vid_index = index
    print(results.keys())
    return results

def default_cfg():
    # -- config --
    cfg = edict()
    cfg.nframes = 5
    cfg.checkpoint_dir = "/home/gauenk/Documents/packages/colanet/output/checkpoints/"
    cfg.num_workers = 4
    cfg.device = "cuda:0"
    cfg.batch_size = 1
    cfg.saved_dir = "./output/saved_results/"
    cfg.device = "cuda:0"
    cfg.dname = "davis"
    cfg.flow = "true"
    cfg.mtype = "gray"
    cfg.bw = True
    cfg.nsamples_at_testing = 10
    cfg.nsamples_tr = 500
    cfg.nsamples_val = 30
    cfg.rand_order_val = False
    cfg.index_skip_val = 5
    cfg.nepochs = 5
    cfg.ensemble = "false"
    cfg.log_root = "./output/bwd_error_map/log"
    return cfg

def main():
    # -- print os pid --
    print("PID: ",os.getpid())

    # -- init --
    verbose = True
    cache_name = "bwd_error_map"
    cache = cache_io.ExpCache(".cache_io",cache_name)
    # cache.clear()

    # -- create exp list --
    ws,wt,ps,nchnls = [10],[5],[7],[16]
    nbatch,k = [1024],[100]
    sigmas = [30.]
    isizes = ["128_128"]
    exact = ["false"]
    ca_fwd_list = ["dnls_k"]
    rep_ids = list(np.arange(3))
    seeds = list(np.arange(100))
    indices = list(np.linspace(0,30,6).astype(np.int32))
    exp_lists = {"sigma":sigmas,"ws":ws,"wt":wt,"isize":isizes,
                 "ca_fwd":ca_fwd_list,"seed":seeds,"rep_id":rep_ids,
                 "sample_index":indices,"nbatch":nbatch,"nchnls":nchnls,
                 "ps":ps,"k":k}
    exps = cache_io.mesh_pydicts(exp_lists) # create mesh
    nexps = len(exps)

    # -- group with default --
    cfg = default_cfg()
    cache_io.append_configs(exps,cfg) # merge the two

    # -- launch each experiment --
    for exp_num,exp in enumerate(exps):

        # -- info --
        if verbose:
            print("-="*25+"-")
            print(f"Running experiment number {exp_num+1}/{nexps}")
            print("-="*25+"-")
            pp.pprint(exp)

        # -- logic --
        uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid
        # cache.clear_exp(uuid)
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_experiment(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- results --
    path = Path("./output/bwd_error_map/records.pkl")
    records = cache.load_flat_records(exps,path,clear=True)
    print(records)

    # -- compare to gt --


if __name__ == "__main__":
    main()

