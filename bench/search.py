
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
    th.cuda.empty_cache()
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

def run_prod_search(rec,fflow,bflow,
                    k,ps,ws,wt,nheads,
                    stride0,stride1,
                    anchor_self,
                    b,t,c,h,w,device):
    # -- misc --
    name = "prod_search"

    # -- init search params --
    vid0 = th.rand((b,t,nheads*c,h,w),device=device)
    vid1 = th.rand((b,t,nheads*c,h,w),device=device)
    fflow = th.rand((b,t,2,h,w),device=device)
    bflow = th.rand((b,t,2,h,w),device=device)
    pt = 1
    use_k = k > 0

    search = dnls.search.init("prod_search_with_heads",
                              fflow, bflow,
                              k, ps, pt, ws, wt, nheads,
                              chnls=-1,stride0=stride0, stride1=stride1,
                              anchor_self=anchor_self,use_self=anchor_self,
                              use_k=use_k)
    # -- burn-in --
    dists,inds = search(vid0,vid1)
    th.cuda.synchronize()
    th.cuda.empty_cache()

    # -- entire search --
    with rec(name,True):
        dists,inds = search(vid0,vid1)

def run_prod_search_v2(rec,fflow,bflow,
                       k,ps,ws,wt,nheads,
                       stride0,stride1,anchor_self,
                       b,t,c,h,w,device):
    # -- misc --
    th.cuda.empty_cache()
    name = "refactored_search"

    # -- init search params --
    vid0 = th.rand((b,t,nheads*c,h,w),device=device)
    vid1 = th.rand((b,t,nheads*c,h,w),device=device)
    fflow = th.rand((b,t,2,h,w),device=device)
    bflow = th.rand((b,t,2,h,w),device=device)
    search = dnls.search.init("search_with_heads",
                              ws, wt, ps, k, nheads,
                              stride0=stride0, stride1=stride1,
                              anchor_self=anchor_self)

    # -- burn-in --
    dists,inds = search(vid0,vid1,fflow,bflow)
    th.cuda.synchronize()
    th.cuda.empty_cache()

    # -- iqueries --
    with rec(name,True):
        dists,inds = search(vid0,vid1,fflow,bflow)

def main():


    # -- init recording --
    timer = ExpTimer()
    gpu_rec = GpuRecord()
    rec = RecordIt(gpu_rec,timer)

    # -- params --
    fflow,bflow = None,None
    nheads = 3
    k,ps,pt = 10,10,1
    ws,wt = 15,5
    dil,stride0,stride1 = 1,4,1
    b,t,c,h,w = 1,5,3,512,512
    device = "cuda:0"
    anchor_self = False

    # -- comparisons --
    # run_l2_search(rec,fflow,bflow,
    #               k,ps,pt,ws,wt,dil,
    #               stride0,stride1,
    #               t,c,h,w,device)
    # run_l2_search_with_index(rec,fflow,bflow,
    #                          k,ps,pt,ws,wt,dil,
    #                          stride0,stride1,
    #                          t,c,h,w,device)
    # print(rec)

    run_prod_search_v2(rec,fflow,bflow,k,ps,ws,wt,nheads,stride0,
                       stride1,anchor_self,b,t,c,h,w,device)
    run_prod_search(rec,fflow,bflow,k,ps,ws,wt,nheads,stride0,
                    stride1,anchor_self,b,t,c,h,w,device)
    print(rec)


if __name__ == "__main__":
    main()
