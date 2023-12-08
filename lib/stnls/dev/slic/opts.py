

import torch as th
import stnls
from einops import rearrange,repeat

def graph_transpose_q2k(dists_k,flows_k,flows,ws,wt,stride0,H,W,full_ws):

    # -- create scattering labels for graph transpose [aka *magic*] --
    names,labels = stnls.graph_opts.scatter_labels(flows,flows_k,ws,wt,
                                                   stride0,1,H,W,full_ws)

    # -- create unique scatter indices [the *magic*] --
    stride1 = 1
    B,HD,Q,S = labels.shape
    gather_labels = repeat(th.arange(S),'s -> b hd q s',b=B,hd=HD,q=Q).int()
    gather_labels = gather_labels.reshape_as(dists_k).to(dists_k.device)

    # -- scattering top-K=1 --
    scatter_weights = stnls.graph_opts.scatter_tensor(dists_k,flows_k,labels,
                                               stride0,stride1,H,W)
    scatter_flows_k = stnls.graph_opts.scatter_tensor(flows_k,flows_k,labels,
                                               stride0,stride1,H,W)
    scatter_labels = stnls.graph_opts.scatter_tensor(gather_labels,flows_k,labels,
                                              stride0,stride1,H,W,invalid=-th.inf)
    scatter_flows_k = -scatter_flows_k

    return scatter_weights,scatter_flows_k,scatter_labels


def graph_transpose_k2q(s_dists,s_flows,s_labels,stride0,T,H,W):


    # -- prepare weights and flows --
    B,HD,_,K0 = s_dists.shape
    s_dists = s_dists.reshape(B,HD,T,H,W,K0)
    s_flows = s_flows.reshape(B,HD,T,H,W,K0,3)
    s_labels = s_labels.reshape(B,HD,T*H*W,-1)

    # -- scatter --
    stride1 = 1
    # dists = stnls.graph_opts.gather_tensor(s_dists,s_flows,s_labels,
    #                                        stride0,stride1,H,W)
    # flows = stnls.graph_opts.gather_tensor(s_flows,s_flows,s_labels,
    #                                        stride0,stride1,H,W)
    dists = stnls.graph_opts.scatter_tensor(s_dists,s_flows,s_labels,
                                            stride1,stride0,H,W)
    flows = stnls.graph_opts.scatter_tensor(s_flows,s_flows,s_labels,
                                            stride1,stride0,H,W)

    # flows = -flows

    # -- reshape --
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    dists = dists.reshape(B,HD,T,nH,nW,-1)
    flows = flows.reshape(B,HD,T,nH,nW,-1,3)

    return dists,flows

