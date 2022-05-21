

import torch as th

def get_inorder_batch(batch,bsize,t,h,w,device="cpu"):

    # -- derivatives --
    hw = h*w
    npix = t*h*w

    # -- start and end --
    start = batch*bsize
    end = min((batch+1)*bsize,npix)

    # -- create raveled --
    qInds = th.arange(start,end,device=device,dtype=th.int32)
    qInds = unravel_index(qInds, (t,h,w))

    return qInds

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = th.div(index,dim,rounding_mode="floor")
    out = list(reversed(out))
    out = th.stack(out,-1)
    return out
