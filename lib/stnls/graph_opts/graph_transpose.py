"""

   Transpose a directed graph
   defined by the distsances and flows from
   the non-local search

"""

class WrapGraph():
    def __init__(self,dists,flows_k,flows,ws,wt,stride0,stride1,H,W,full_ws):
        self.dists = dists
        self.flows_k = flows_k
        self.flows = flows
        names,labels = stnls.agg.scatter_labels(flows,flows_k,ws,wt,
                                                stride0,1,H,W,full_ws)
        self.names = names
        self.labels = labels

    def transpose(self):
        # -- [transpose graph] queries to keys --
        args = ["dists_k","flows_k","flows","labels","ws","wt","stride0","H","W","full_ws"]
        args = [getattr(self,arg) for arg in args]
        return run_scatter_q2k(*args)

    def inv_transpose(self,D,L):
        args = ["flows","labels","ws","wt","stride0","H","W","full_ws"]
        args = [D,L,] + [getattr(self,arg) for arg in args]
        return run_gather_k2q(*args)

    def normalize_searched(self):
        Dt,Ft,Lt = self.transpose()
        D,F,L = run_topk(Dt,Ft,Lt)
        return self.inv_transpose(D,F,L)

def graph_transpose(D,F,L):

        # -- [transpose graph] queries to keys --
        outs = run_scatter_q2k(dists_k,flows_k,flows,labels,ws,wt,stride0,H,W,full_ws)
        scatter_dists,scatter_flows,scatter_labels = outs

        # -- top-k --
        print(scatter_labels.shape,scatter_labels.max(),scatter_labels.min())
        topk,K0 = stnls.agg.scatter_topk,-1
        s_dists,s_flows,s_labels = topk(scatter_dists,scatter_flows,
                                        scatter_labels,K0,descending=False)

        print(s_labels.shape,s_labels.max(),s_labels.min())

        # -- [transpose graph] keys to queries --
        dists_k,flows_k = run_scatter_k2q(s_dists,s_flows,s_labels,stride0,T,H,W)
        print(_d)
        print(dists_k)
        print(_d.shape,dists_k.shape)

    pass

def graph_reverse(D,F,L):
    pass

