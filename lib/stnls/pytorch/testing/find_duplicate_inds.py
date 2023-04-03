
import torch as th

def run(inds):
    """
    inds.shape = [...,K,3]

    """

    # -- reshape --
    oshape = list(inds.shape[:-2])
    K,_ = inds.shape[-2:]
    inds = inds.reshape(-1,K,3).contiguous()
    Q,K,_ = inds.shape

    # -- check diffs --
    dups = th.zeros((Q,K,K),device=inds.device,dtype=th.bool)
    for ki in range(K):
        for kj in range(ki+1,K):
            diff_ij = th.sum(th.abs(inds[:,ki] - inds[:,kj]),-1)
            dups[:,ki,kj] = diff_ij == 0

    # -- report any --
    dups = dups.any(-1).any(-1)
    any_non_uniq = th.any(dups).item()

    # -- shape back --
    dshape = oshape
    dups = dups.reshape(dshape)

    return dups,any_non_uniq

