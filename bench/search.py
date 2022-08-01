
# -- misc --
import dnls
import torch as th

# -- benchmarking imports --
from dnls.utils.timer import ExpTimer
from dnls.utils.bench import RecordIt
from dnls.utils.gpu_mem import GpuRecord
from dnls.utils.inds import get_batching_info

def run_l2_search(rec,fflow,bflow,
                  k,ps,pt,ws,wt,dil,
                  stride0,stride1,
                  t,c,h,w,device):
    # -- misc --
    name = "l2_search"

    # -- init search params --
    vid0 = th.rand((t,c,h,w),device=device)
    vid1 = th.rand((t,c,h,w),device=device)
    search = dnls.search.init("l2",fflow, bflow, k, ps, pt, ws, wt,
                              dilation=dil, stride=stride1)
    ntotal,_,_,_ = get_batching_info(vid0.shape,stride0,stride1,ps,dil)
    nbatch,qindex = ntotal,0

    # -- entire search --
    with rec(name,True):
        qinds = dnls.utils.inds.get_query_batch(qindex,nbatch,stride0,
                                                t,h,w,device)
        dists,inds = search(vid0,qinds,vid1)

    # -- only search --
    with rec(name + "_only_search",True):
        dists,inds = search(vid0,qinds,vid1)

def run_l2_search_with_index(rec,fflow,bflow,
                             k,ps,pt,ws,wt,dil,
                             stride0,stride1,
                             t,c,h,w,device):
    # -- misc --
    name = "l2_search_with_index"

    # -- init search params --
    vid0 = th.rand((t,c,h,w),device=device)
    vid1 = th.rand((t,c,h,w),device=device)
    search = dnls.search.init("l2_with_index",fflow, bflow, k, ps, pt, ws, wt,
                              dilation=dil, stride0=stride0, stride1=stride1)
    ntotal,_,_,_ = get_batching_info(vid0.shape,stride0,stride1,ps,dil)
    nbatch,qindex = ntotal,0

    # -- iqueries --
    with rec(name,True):
        dists,inds = search(vid0,qindex,ntotal,vid1)

def main():


    # -- init recording --
    timer = ExpTimer()
    gpu_rec = GpuRecord()
    rec = RecordIt(gpu_rec,timer)

    # -- params --
    fflow,bflow = None,None
    k,ps,pt = 10,10,1
    ws,wt = 15,5
    dil,stride0,stride1 = 1,4,4
    t,c,h,w = 1,3,512,512
    device = "cuda:0"

    # -- comparisons --
    run_l2_search(rec,fflow,bflow,
                  k,ps,pt,ws,wt,dil,
                  stride0,stride1,
                  t,c,h,w,device)
    run_l2_search_with_index(rec,fflow,bflow,
                             k,ps,pt,ws,wt,dil,
                             stride0,stride1,
                             t,c,h,w,device)
    print(rec)

if __name__ == "__main__":
    main()
