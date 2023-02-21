
import torch as th
import dnls_cuda

# void topk_pwd_forward_cuda(const torch::Tensor vid,
#     const torch::Tensor inds0, const torch::Tensor inds1,
#     torch::Tensor dists, int ps, int pt,
#     int dilation, bool reflect_bounds, bool use_adj,
#     int off_H0, int off_W0, int off_H1, int off_W1){

def run(vid,inds0,inds1,ps,pt=1,dilation=1,
        reflect_bounds=True,use_adj=False,
        off_H0=0,off_W0=0,off_H1=0,off_W1=0):

    # -- allocate --
    K = inds0.shape[-2]
    shape = list(inds0.shape[:-2]) + [int(K*(K+1)/2),]
    dists = th.zeros(shape,dtype=vid.dtype,device=vid.device)

    # -- run --
    dnls_cuda.topk_pwd(vid,inds0,inds1,dists,ps,pt,dilation,
                       reflect_bounds,use_adj,
                       off_H0,off_W0,off_H1,off_W1)
    dists = th.sqrt(dists)

    return dists
