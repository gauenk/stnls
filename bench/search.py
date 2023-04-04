
# -- misc --
import torch as th
import stnls

# -- data mngment --
from easydict import EasyDict as edict

# -- benchmarking imports --
from stnls.utils.timer import ExpTimer
from stnls.utils.bench import RecordIt
from stnls.utils.gpu_mem import GpuRecord
from stnls.utils.inds import get_batching_info

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

def run_search(rec,cfg):

    # -- misc --
    th.cuda.empty_cache()

    # -- init search params --
    vid0,vid1,fflow,bflow = init_data(cfg)
    esearch = stnls.search.non_local_search.init(cfg)
    search = stnls.search.init(cfg)

    # -- exact search --
    dists_e,inds_e = esearch(vid0,vid1,fflow,bflow)
    search = stnls.search.utils.search_wrap(cfg.search_name,search)

    # -- burn-in --
    dists,inds = search(vid0,vid1,fflow,bflow,inds_e,None,None)
    th.cuda.synchronize()
    th.cuda.empty_cache()

    # -- iqueries --
    with rec(cfg.search_name,True):
        dists,inds = search(vid0,vid1,fflow,bflow,inds_e,None,None)

def main():


    # -- init recording --
    timer = ExpTimer()
    gpu_rec = GpuRecord()
    rec = RecordIt(gpu_rec,timer)

    # -- params --
    cfg = {"k":10,"ps":7,"pt":1,"ws":21,"wt":3,"wr":1,"kr":-1,
           "wr_t":1,"kr_t":0.15,"wr_s":1,"kr_s":-1,"scale":4,
           "nheads":3,"dil":1,"stride0":4,"stride1":1,
           "batchsize":1,"nframes":3,"nftrs_per_head":9,
           "height":512,"width":512,"device":"cuda:0","dist_type":"l2"}
    cfg = edict(cfg)
    # search_names = ["nls","refine","approx_t","approx_s","approx_st"]
    search_names = ["nls","approx_s"]
    for search_name in search_names:
        cfg["search_name"] = search_name
        run_search(rec,cfg)
    print(rec)


if __name__ == "__main__":
    main()
