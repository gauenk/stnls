
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
    # softmax_weight = 1

    # -- init video --
    vid = append_grid(vid,M/stride0)
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
    dists_k = th.ones((B,HD,T,nH,nW,9),device=device,dtype=th.float32)
    flows_k = th.zeros((B,HD,T,nH,nW,9,3),device=device,dtype=th.int)
    for i in range(3):
        for j in range(3):
            k = j + i*3
            flows_k[...,k,1] = i-1
            flows_k[...,k,2] = j-1

    # -- pooling (u_c) --
    agg_ps = 1
    weights = th.softmax(-softmax_weight*dists_k,-1)
    agg = stnls.agg.NonLocalGatherAdd(agg_ps,stride0,1,
                                      outH=nH,outW=nW,itype="int")
    pooled = rearrange(agg(vid,weights,flows_k),'b hd t c h w -> b t (hd c) h w')
    pooled_prev = pooled
    # print("Delta: ",th.mean((pooled-pooled0)**2).item())
    # print("[gather] pooled.shape: ",pooled.shape)
    print("[gather] pooled.shape: ",pooled.shape)

    # inds = stnls.utils.misc.flow2inds(flows_k,stride0).long()
    # # inds = rearrange(inds,'b hd t h w 1 tr -> (b hd) (t h w) tr')
    # # print(inds[...,1].max(),inds[...,2].max(),inds[...,1].min(),inds[...,2].min(),H,W)
    # s_dists,s_flows = dists_k,flows_k

    # -- modified search space [get btm-right edges] --
    ws_og = ws
    # print(H-((nH-1)*stride0+(ws_og-(ws_og-1)//2)),0)
    # print(H,(nH-1)*stride0,ws_og,(ws_og-1)//2)
    extra = max(H-((nH-1)*stride0+(ws_og-(ws_og-1)//2)),0)
    ws = ws+extra
    # ws_og = ws
    # print(ws_og,ws,extra,H-(nH-1)*stride0,stride0,nH,H)
    # exit()

    # -- iterations --
    assert niters > 0
    for i in range(niters):


        #
        # -- compute distances --
        #

        # -- compute pairwise searches --
        full_ws = False
        k = ws*ws*(2*wt+1)
        # k = -1
        search = stnls.search.NonLocalSearch(ws,wt,ps,k,nheads=1,dist_type="l2",
                                             stride0=stride0,strideQ=1,
                                             ws_interior=ws_og,
                                             self_action="anchor_self",
                                             # reflect_bounds=False,
                                             full_ws=full_ws,itype="int")
        dists_k,flows_k = search(pooled,vid,flows)
        # print("dists_k.shape: ",dists_k.shape)
        # print(dists_k[0,0,0,0,0])
        # exit()

        # valid = th.where(th.abs(flows_k[...,0])<1e3)
        # print(flows_k[...,0][valid].unique())
        # print(th.all(flows_k[...,0][valid] == 0))
        # exit()

        # dists_k = th.exp(-softmax_weight*dists_k) # normalize
        # dists_k = th.softmax(-softmax_weight*dists_k,-1) # normalize
        _d,_f = dists_k,flows_k
        # print(dists_k.shape)
        # print("[pre]: ",dists_k[0,0,0,:2,:2,:10])

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
        # # # print(s_dists)
        # print("s_dists.shape: ",s_dists.shape)
        # print("s_flows.shape: ",s_flows.shape)
        s_dists_rs = s_dists.reshape(T,H,W,-1)
        s_flows_rs = s_flows.reshape(T,H,W,-1,3)
        # print("s_dists.shape: ",s_dists.shape)
        print("-"*20)
        print("-"*20)
        print("-"*20)
        print(s_dists_rs[0,H//2+1,29])
        print(s_flows_rs[0,H//2+1,29])
        print("-"*20)
        print(s_dists_rs[0,H//2,29])
        print(s_flows_rs[0,H//2,29])
        print("-"*20)
        print(s_dists_rs[0,H//2-1,29])
        print(s_flows_rs[0,H//2-1,29])
        print("-"*20)
        print(s_dists_rs[0,H//2+2,29])
        print(s_flows_rs[0,H//2+2,29])


        print("-"*20)
        print("-"*20)
        print("-"*20)
        print(s_dists_rs[0,H//2+6,29])
        print(s_flows_rs[0,H//2+6,29])
        print("-"*20)
        print(s_dists_rs[0,H//2+7,29])
        print(s_flows_rs[0,H//2+7,29])
        print("-"*20)
        print(s_dists_rs[0,H//2+5,29])
        print(s_flows_rs[0,H//2+5,29])
        print("-"*20)
        print(s_dists_rs[0,H//2+8,29])
        print(s_flows_rs[0,H//2+8,29])

        # exit()

        # # print(s_dists[0,0,50:60])
        # s_dists = th.softmax(-softmax_weight*s_dists,-1) # normalize
        # # # # s_dists = th.exp(-softmax_weight*s_dists) # normalize

        # # -- [transpose graph] keys to queries --
        # dists_k,flows_k = graph_transpose_k2q(s_dists,s_flows,s_labels,stride0,T,H,W)
        # print("[post]: ",dists_k[0,0,0,:2,:2,:10])
        # # valid = th.logical_and(th.abs(_d) < 200,th.abs(dists_k) < 200)
        # # print(dists_k.shape,_d.shape)

        #
        # -- reconstruct from all neighbors (u_c) --
        #

        dists_k = th.softmax(-softmax_weight*dists_k,-1) # normalize
        # dists_k = th.softmax(dists_k,-1) # normalize
        # dists_k = dists_k/dists_k.sum(-1,keepdim=True)

        assert th.any(th.isinf(dists_k)).item() is False,"Any invalid?"
        # weights = th.softmax(-softmax_weight*dists_k,-1)
        print("[post]: ",dists_k[0,0,0,:2,:2,:10])
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
        print("[scatter] pooled.shape: ",pooled.shape,pooled.max().item())

        # -- ave non-empty cluster size --
        cz = (dists_k>0).float().sum(-1)
        ave_cz = th.mean(cz[th.where(cz>0)]).item()

        # -- score --
        # print(dists_k[0,0,0,:2,:2,:10])
        print("Average Cluster Size: ",ave_cz)
        print("Reconstruct: ",th.mean((vid-reconstruct)**2).item())
        print("Score: ",th.mean((pooled_prev-pooled)**2).item())
        # print("Score: ",compute_score(vid,pooled,flows_k,ws,wt,ps,stride0,stride0))
        pooled_prev = pooled

    return pooled,dists_k,flows_k,s_dists,s_flows
