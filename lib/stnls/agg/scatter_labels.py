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

def run(flows,flows_k,ws,wt,stride0,stride1,full_ws):

    # -- unpack shapes --
    B,HD,T,nH,nW,K,_ = flows_k.shape
    # B,HD,T,W_t,2,nH,nW = flows.shape
    Q = T*nH*nW
    W_t = 2*wt+1
    H = nH*stride0
    W = nW*stride0

    # -- number of maximum possible groups a single patch can belong to --
    print(full_ws)
    if not(full_ws):
        S = W_t*ws*ws
    else:
        # wsHalf = ws
        Ws_num = ws*ws + ws*ws#wsHalf*wsHalf
        S = W_t*Ws_num

    # -- prepare --
    labels = -th.ones((B,HD,Q,K),device=flows.device,dtype=th.int)
    names = -th.ones((B,HD,S,T,H,W,2),device=flows.device,dtype=th.int)
    print(flows.shape,flows_k.shape,names.shape,labels.shape)

    # -- fill init labels --
    stnls_cuda.scatter_labels(flows,flows_k,labels,names,ws,wt,stride0,stride1,full_ws)

    # -- check --
    nvalid = (names[...,0] >= 0).float().sum(2)
    print(nvalid.sum(),Q*K)
    print(th.all(nvalid>0))
    # assert th.all(nvalid>0)
    print(nvalid.max())
    # print(nvalid)

    print("")
    print("-"*10 + "< names >" + "-"*10)
    print("")

    print("-"*5)
    print(names[0,0,:,0,0,0].T)
    print(names[0,0,:,0,1,1].T)
    print(names[0,0,:,0,2,2].T)
    # print(names[0,0,:,0,3,3])

    print("-"*5)
    print(names[0,0,0,0,:,:,0])
    print(names[0,0,0,0,:,:,1])
    print("-"*5)
    print(names[0,0,1,0,:,:,0])
    print(names[0,0,1,0,:,:,1])
    print("-"*5)
    print(names[0,0,4,0,:,:,0])
    print(names[0,0,4,0,:,:,1])


    # -- names is correct --
    print("")
    print("-"*5 + "< [names] iterating >" + "-"*5)
    Q = T*nH*nW
    print(names[...,0].max().item(),names[...,1].max().item(),Q,S)
    print((names>=0).sum(),Q*K)
    for i in range(Q):
        for j in range(S):
            check0 = names[...,0]==i
            check1 = names[...,1]==j
            check = check0 * check1
            num = check.sum().item()
            # print("any %d,%d?: "%(i,j),check.sum().item())
            if num > 1: print("invalid %d,%d?: %d"%(i,j,num))
    print("")

    # -- labels --
    # labels = names2labels(names,Q,S)
    print("")
    print("-"*10 + "< labels >" + "-"*10)
    print("")
    valid0 = labels >= -1
    valid1 = labels <= (S-1)
    valid = th.all(valid0 * valid1).item()
    print("All valid labels? ",valid)

    print(labels[0,0,0,:])
    print(labels[0,0,1,:])
    print(labels[0,0,-3:])


    return labels


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
