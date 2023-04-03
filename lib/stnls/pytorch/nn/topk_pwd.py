
import torch as th
import stnls_cuda

def run(vid,inds0,inds1,ps,pt=1,dilation=1,
        reflect_bounds=True,use_adj=False,
        off_H0=0,off_W0=0,off_H1=0,off_W1=0):

    # -- allocate --
    K = inds0.shape[-2]
    shape = list(inds0.shape[:-2]) + [int(K*(K-1)/2),]
    dists = th.zeros(shape,dtype=vid.dtype,device=vid.device)

    # -- run --
    stnls_cuda.topk_pwd(vid,inds0,inds1,dists,ps,pt,dilation,
                       reflect_bounds,use_adj,
                       off_H0,off_W0,off_H1,off_W1)
    dists = th.sqrt(dists)

    return dists
