

import torch as th

def run_batched(run_fxn,batchsize,ntotal,nbatches,*args):
    dists,inds = [],[]
    for batch in range(nbatches):
        qshift = batch*batchsize
        nqueries = min(ntotal-qshift,batchsize)
        # print(nbatches,batch,qshift,nqueries,ntotal)
        assert nqueries > 0
        dists_b,inds_b = run_fxn(qshift,nqueries,*args)
        dists.append(dists_b)
        inds.append(inds_b)
    dists = th.cat(dists,2)
    inds = th.cat(inds,2)
    return dists,inds

def batching_info(vid,stride0,ws,wt,batchsize):

    # -- compute num refs --
    B,HD,T,C,H,W = vid.shape
    nH = (H-1)//stride0+1
    nW = (W-1)//stride0+1
    ntotal = T * nH * nW

    # -- recompute batch size w.r.t max size --
    batchsize = get_max_batchsize(batchsize,ntotal,ws,wt)


    nbatches = (ntotal-1)//batchsize+1
    return ntotal,nbatches,batchsize

def get_max_batchsize(batchsize,nrefs,ws,wt):
    # ntotal_locs = nrefs * nsearch
    # ntotal_ints = ntotal_locs*ntotal_search
    st = 2 * wt + 1
    nsearch = ws * ws * st
    # nmax = 2**31-1
    nmax = 2**22
    max_nrefs = int(nmax / (nsearch*3))
    # print(batchsize,max_nrefs,nrefs,ws,wt,nsearch)
    if batchsize <= 0:
        batchsize = min(max_nrefs,nrefs)
    batchsize = min(batchsize,min(max_nrefs,nrefs))
    return batchsize
