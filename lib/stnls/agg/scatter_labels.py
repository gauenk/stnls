"""

  Compute the destination indices for NonLocalScatter
  to remove the race condition of center
  and mitigate the race condition of the patch

 flows_k MUST be output from a grid search of size (2*wt+1,ws,ws)
 offset from flows. This is not valid for aribtrary values of flows_k

"""


# -- imports --
import torch as th

# -- cpp cuda kernel --
import stnls_cuda

def run(flows,flows_k,ws,wt,stride0,stride1,H,W,full_ws):

    """

    There is a minimum and maximum (ws) depending on (stride0)

    - [max] we don't want overlap of query points
    - [min] we don't want skipped key points

    """

    # -- unpack shapes --
    B,HD,T,nH,nW,K,_ = flows_k.shape
    # B,HD,T,W_t,2,nH,nW = flows.shape
    Q = T*nH*nW
    W_t = 2*wt+1
    # H = nH*stride0
    # W = nW*stride0
    # wsHalf = (ws-1)//2

    # -- number of maximum possible groups a single patch can belong to --
    Wt_num = T if wt > 0 else 1
    # Ws_num = ws*ws
    wsNum = (ws)//stride0+1
    Ws_num = wsNum*wsNum
    if full_ws: Ws_num += 2*wsNum*(wsNum//2) + (wsNum//2)**2
    S = Wt_num*Ws_num
    # print(S,ws,wt,stride0,stride1,full_ws)

    # -- prepare --
    labels = -th.ones((B,HD,Q,K),device=flows.device,dtype=th.int)
    names = -th.ones((B,HD,S,T,H,W,2),device=flows.device,dtype=th.int)
    # print(flows.shape,flows_k.shape,names.shape,labels.shape)

    # -- fill init labels --
    stnls_cuda.scatter_labels(flows,flows_k,labels,names,ws,wt,stride0,stride1,full_ws)

    # -- check --
    nvalid = (names[...,0] >= 0).float().sum(2)
    if full_ws:
        assert(int(nvalid.sum().item()) == Q*K)

    return names,labels


def names2labels(names,Q,S):
    B,HD,S,T,H,W,_ = names.shape
    labels = -th.ones((B,HD,Q,S),dtype=th.int,device=names.device)
    for ibatch in range(B):
        for ihead in range(HD):
            for ti in range(T):
                for hi in range(H):
                    for wi in range(W):
                        for si in range(S):
                            qi,ki = names[ibatch,ihead,si,ti,hi,wi]
                            if (qi<0): continue
                            if qi == 0: print(qi,ki)
                            labels[ibatch,ihead,qi,ki] += 1
    return labels
