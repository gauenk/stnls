

import torch as th

def run_batched(run_fxn,batchsize,vid_idx,stride0_idx):
    dists,inds = [],[]
    ntotal,nbatches = batching_info(args[vid_idx],args[stride0_idx],batchsize)
    for batch in range(nbatches):
        qshift = nbatch*batchsize
        nqueries = min(ntotal-qshift,batchsize)
        assert nqueries > 0
        dists_b,inds_b = run_fxn(qshift,nqueries,*args)
        dists.append(dists_b)
        inds.append(inds_b)
    dists = th.stack(dists,2)
    inds = th.stack(inds,2)
    return dists,inds

def batching_info(vid,stride0,batchsize):
    B,HD,T,C,H,W = vid.shape
    nH = (H-1)//stride0+1
    nW = (W-1)//stride0+1
    ntotal = T * nH * nW
    nbatches = (ntotal-1)//batchsize+1
    return ntotal,nbatches
