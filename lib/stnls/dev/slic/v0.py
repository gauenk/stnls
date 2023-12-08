
# -- basic --
import torch as th
import stnls
from dev_basics import flow
from einops import rearrange,repeat

# -- local imports --
from .opts import graph_transpose_k2q,graph_transpose_q2k
from .utils import append_grid

def run_slic(vid,ws,wt,ps,stride0,full_ws,M=0.5,
             softmax_weight=10.,niters=1,use_rand=False):

    # -- config --
    use_flow = False
    ps = 1
    agg_ps = ps

    # -- init video --
    vid = append_grid(vid,M,stride0)
    B,T,F,H,W = vid.shape

    # -- init flows --
    flows = flow.orun(vid,use_flow,ftype="cv2")
    flows = stnls.nn.search_flow(flows.fflow,flows.bflow,wt,stride0)
    flows = flows[:,None].round().int()

    # -- init slic state --
    device = vid.device
    HD = 1
    H,W = vid.shape[-2:]
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    dists_k = th.ones((B,HD,T,nH,nW,1),device=device,dtype=th.float32)
    flows_k = th.zeros((B,HD,T,nH,nW,1,3),device=device,dtype=th.int)

    # -- pooling (u_c) --
    weights = th.softmax(-softmax_weight*dists_k,-1)
    agg = stnls.agg.NonLocalGatherAdd(agg_ps,stride0,1,
                                      outH=nH,outW=nW,itype="int")
    pooled = rearrange(agg(vid,weights,flows_k),'b hd t c h w -> b t (hd c) h w')
    # print("Delta: ",th.mean((pooled-pooled0)**2).item())
    # print("[gather] pooled.shape: ",pooled.shape)

    # inds = stnls.utils.misc.flow2inds(flows_k,stride0).long()
    # # inds = rearrange(inds,'b hd t h w 1 tr -> (b hd) (t h w) tr')
    # # print(inds[...,1].max(),inds[...,2].max(),inds[...,1].min(),inds[...,2].min(),H,W)

    # -- iterations --
    assert niters > 0
    for i in range(niters):


        #
        # -- compute distances --
        #

        # -- compute pairwise searches --
        full_ws = False
        search = stnls.search.NonLocalSearch(ws,wt,ps,-1,nheads=1,dist_type="l2",
                                             stride0=stride0,strideQ=1,
                                             self_action="anchor_self",
                                             full_ws=full_ws,itype="int")
        dists_k,flows_k = search(pooled,vid,flows)
        _d,_f = dists_k,flows_k

        #
        # -- normalizes similarities across each pixel  [q_c(p)] --
        #


        # -- [transpose graph] queries to keys --
        outs = graph_transpose_q2k(dists_k,flows_k,flows,ws,wt,stride0,H,W,full_ws)
        scatter_dists,scatter_flows,scatter_labels = outs

        # -- top-k --
        topk,K0 = stnls.graph_opts.scatter_topk,-1
        s_dists,s_flows,s_labels = topk(scatter_dists,scatter_flows,
                                        scatter_labels,K0,descending=False)
        s_dists = th.softmax(-softmax_weight*s_dists,-1) # normalize

        # -- [transpose graph] keys to queries --
        dists_k,flows_k = graph_transpose_k2q(s_dists,s_flows,s_labels,stride0,T,H,W)
        # valid = th.logical_and(th.abs(_d) < 200,th.abs(dists_k) < 200)
        print(dists_k.shape,_d.shape)

        #
        # -- reconstruct from all neighbors (u_c) --
        #

        # weights = th.softmax(-softmax_weight*dists_k,-1)
        weights = dists_k
        # print("pooled.shape: ",pooled.shape)
        agg = stnls.agg.NonLocalScatterAdd(agg_ps,1,stride0,outH=H,outW=W,itype="int")
        # agg = stnls.agg.PooledPatchSum(agg_ps,stride0,itype="int")
        shape_str = 'b hd t c h w -> b t (hd c) h w'
        reconstruct = rearrange(agg(pooled,weights,flows_k),shape_str)
        # print("Delta: ",th.mean((pooled-pooled0)**2).item())
        print("[scatter] reconstruct.shape: ",reconstruct.shape)

        #
        # -- pool from all neighbors (u_c) --
        #

        # weights = th.softmax(-softmax_weight*dists_k,-1)
        weights = dists_k
        agg = stnls.agg.NonLocalGatherAdd(agg_ps,stride0,1,outH=nH,outW=nW,itype="int")
        shape_str = 'b hd t c h w -> b t (hd c) h w'
        pooled = rearrange(agg(vid,weights,flows_k),shape_str)
        print("[scatter] pooled.shape: ",pooled.shape)


        # -- score --
        print("Score: ",th.mean((vid-reconstruct)**2).item())
        # print("Score: ",compute_score(vid,pooled,flows_k,ws,wt,ps,stride0,stride0))

    return pooled,dists_k,flows_k,s_dists,s_flows
