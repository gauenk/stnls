
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

def run_agg(rec,cfg):

    # -- misc --
    th.cuda.empty_cache()

    # -- run search --
    vid0,vid1,fflow,bflow = init_data(cfg)
    search = stnls.search.non_local_search.init(cfg)
    dists,inds = search(vid0,vid1,fflow,bflow)

    # -- aggregate --
    agg = stnls.reducer.init(cfg)

    # -- iqueries --
    with rec(cfg.reducer_name,True):
        vid = agg(vid0,dists,inds)

def main():


    # -- init recording --
    timer = ExpTimer()
    gpu_rec = GpuRecord()
    rec = RecordIt(gpu_rec,timer)

    # -- params --
    cfg = {"k":10,"ps":7,"pt":1,"ws":9,"wt":3,"wr":1,"kr":-1,
           "wr_t":1,"kr_t":0.15,"wr_s":1,"kr_s":-1,"scale":4,
           "nheads":8,"dil":1,"stride0":4,"stride1":1,
           "batchsize":1,"nframes":5,"nftrs_per_head":9,
           "height":512,"width":512,"device":"cuda:0","dist_type":"l2"}
    cfg = edict(cfg)
    reducer_names = ["wpsum","fwpsum","iwpsum"]
    for reducer_name in reducer_names:
        cfg["reducer_name"] = reducer_name
        run_agg(rec,cfg)
    print(rec)


if __name__ == "__main__":
    main()
