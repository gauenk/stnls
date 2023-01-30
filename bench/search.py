
# -- misc --
import torch as th
import dnls

# -- data mngment --
from easydict import EasyDict as edict

# -- benchmarking imports --
from dnls.utils.timer import ExpTimer
from dnls.utils.bench import RecordIt
from dnls.utils.gpu_mem import GpuRecord
from dnls.utils.inds import get_batching_info

def init_data(cfg):
    B = cfg.batchsize
    T = cfg.nframes
    HD = cfg.nheads
    F_HD = cfg.nftrs_per_head
    H = cfg.height
    W = cfg.width
    device = cfg.device
    vid0 = th.rand((B,T,HD*F_HD,H,W),device=device)
    vid1 = th.rand((B,T,HD*F_HD,H,W),device=device)
    fflow = th.rand((B,T,2,H,W),device=device)
    bflow = th.rand((B,T,2,H,W),device=device)
    return vid0,vid1,fflow,bflow

def search_wrap(name,search):
    if "refine" in name:
        def wrap(vid0,vid1,fflow,bflow,inds):
            return search(vid0,vid1,inds)
        return wrap
    else:
        def wrap(vid0,vid1,fflow,bflow,inds):
            return search(vid0,vid1,fflow,bflow)
        return wrap

def run_search(rec,cfg):

    # -- misc --
    th.cuda.empty_cache()

    # -- init search params --
    vid0,vid1,fflow,bflow = init_data(cfg)
    esearch = dnls.search.non_local_search.init(cfg)
    search = dnls.search.init(cfg)

    # -- exact search --
    dists_e,inds_e = esearch(vid0,vid1,fflow,bflow)
    search = search_wrap(cfg.search_name,search)

    # -- burn-in --
    dists,inds = search(vid0,vid1,fflow,bflow,inds_e)
    th.cuda.synchronize()
    th.cuda.empty_cache()

    # -- iqueries --
    with rec(cfg.search_name,True):
        dists,inds = search(vid0,vid1,fflow,bflow,inds_e)

def main():


    # -- init recording --
    timer = ExpTimer()
    gpu_rec = GpuRecord()
    rec = RecordIt(gpu_rec,timer)

    # -- params --
    cfg = {"k":10,"ps":10,"pt":1,"ws":15,"wt":3,"wr":1,"kr":1,"scale":4,
           "nheads":3,"dil":1,"stride0":4,"stride1":1,
           "batchsize":1,"nframes":3,"nftrs_per_head":9,
           "height":256,"width":256,"device":"cuda:0","dist_type":"prod"}
    cfg = edict(cfg)
    search_names = ["nls","refine","approx_t","approx_s","approx_st"]
    for search_name in search_names:
        cfg["search_name"] = search_name
        run_search(rec,cfg)
    print(rec)


if __name__ == "__main__":
    main()
