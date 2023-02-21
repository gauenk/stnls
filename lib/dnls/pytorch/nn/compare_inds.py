"""

Compute the minimum MSE of sorted patches distances from two sets of indices

"""

import torch as th
# import dnls
import dnls_cuda
from .topk_pwd import run as topk_pwd

def run(vid,inds0,inds1,batchsize=-1):

    # -- compute batches --
    Q = inds0.shape[2]
    if batchsize >= 0:
        nbatches = 1
    else:
        nbatches = Q // batchsize + 1

    # -- run for batches --
    mse = 0
    for batch in range(nbatches):

        # -- limits --
        start = batch*batchsize
        stop = min(start+batchsize,Q)

        # -- batch --
        inds0_b = inds0[:,:,start:stop].contiguous()
        inds1_b = inds1[:,:,start:stop].contiguous()

        # -- exec --
        pwd = topk_pwd(vid,inds0_b,inds1_b)
        print(pwd.shape)
        mse += th.min(pwd,1)

    # -- normalize --
    mse /= nbatches
    return mse



