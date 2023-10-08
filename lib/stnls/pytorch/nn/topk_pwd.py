
import torch as th
import stnls_cuda

def run(vid,inds0,inds1,ps,pt=1,dilation=1,
        reflect_bounds=True,use_adj=False):

    # -- allocate --
    K = inds0.shape[-2]
    shape = list(inds0.shape[:-2]) + [int(K*(K-1)/2),]
    dists = th.zeros(shape,dtype=vid.dtype,device=vid.device)

    # -- run --
    patch_offset = 0 if use_adj else -ps//2
    stnls_cuda.topk_pwd(vid,inds0,inds1,dists,ps,pt,dilation,
                       reflect_bounds,patch_offset)
    dists = th.sqrt(dists)

    return dists
