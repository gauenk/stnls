"""

  Compute the destination indices for NonLocalScatter
  to remove the race condition of center
  and mitigate the race condition of the patch

"""


# -- imports --
import torch as th

# -- cpp cuda kernel --
import stnls_cuda

def run(flows,flows_k,ws,wt,stride0,stride1,full_ws):

    # -- unpack shapes --
    B,HD,T,nH,nW,K,_ = flows_k.shape
    # B,HD,T,W_t,2,nH,nW = flows.shape
    Q = T*nH*nW
    W_t = 2*wt+1
    S = W_t*ws*ws
    H = nH*stride0
    W = nW*stride0

    # -- prepare --
    labels = th.zeros((B,HD,Q,K),device=flows.device,dtype=th.int)
    names = -th.ones((B,HD,S,T,H,W,2),device=flows.device,dtype=th.int)
    print(flows.shape,flows_k.shape,names.shape,labels.shape)

    # -- fill init labels --
    stnls_cuda.scatter_labels(flows,flows_k,labels,names,ws,wt,stride0,stride1,full_ws)

    return labels
