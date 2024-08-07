

import math
import stnls
import torch as th
import numpy as np
import random


def get_data(ws,ws_og,wt,T,H,W,stride0,stride1,full_ws):

    # -- get args --
    seed = 123
    nheads = 1
    device = "cuda:0"
    set_seed(seed)
    W_t = 2*wt+1
    B,HD,F = 1,1,1
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    # ws_og = ws
    # ws = ws + (H-(nH-1)*stride0)
    # print("ws,ws_og: ",ws,ws_og)
    K = ws*ws*W_t
    itype = "int"
    # K = -1

    # -- load flows --
    flows = th.ones((B,HD,T,W_t-1,2,nH,nW)).cuda()/2.
    flows = th.rand_like(flows)/2.+th.randint_like(flows,-3,3)+0.2
    flows = th.zeros_like(flows)
    flows = flows.round().int()

    # -- load top-k flows --
    ps = 3
    search = stnls.search.NonLocalSearch(ws, wt, ps, K, nheads, dist_type="l2",
                                         stride0=stride0, stride1=stride1,
                                         reflect_bounds=True,
                                         self_action="anchor",
                                         ws_interior=ws_og,
                                         itype="int",full_ws=full_ws)
    vid = th.rand((B,HD,T,F,H,W)).to(device)
    _,flows_k = search(vid,vid,flows)
    return flows_k[0][0].cpu().numpy()

def check_valid(ti,hi,wi,T,H,W):
    valid = (ti >= 0) and (ti < T)
    valid = valid and (hi >= 0) and (hi < H)
    valid = valid and (wi >= 0) and (wi < W)
    return valid

def set_search_offsets(wsOff_h, wsOff_w, hi, wi,
                       stride1, wsHalf, ws, H, W, full_ws):
    if full_ws is False: return wsHalf,wsHalf

    # -- bound min --
    if ( (hi - stride1 * wsHalf) < 0):
        wsOff_h = math.floor(hi/(1.*stride1))
    if ( (wi - stride1 * wsHalf) < 0):
        wsOff_w = math.floor(wi/(1.*stride1))

    #  -- bound max --
    hMax = hi + stride1 * ((ws-1) - wsOff_h)
    wMax = wi + stride1 * ((ws-1) - wsOff_w)
    if (hMax > (H-1)):
        wsOff_h = math.ceil((hi - (H-1))/(1.*stride1) + (ws-1));
    if (wMax > (W-1)):
        wsOff_w = math.ceil((wi - (W-1))/(1.*stride1) + (ws-1));

    return wsOff_h,wsOff_w

def vprint(*args,**kwargs):
    verbose = False
    if verbose:
        print(*args,**kwargs)

def get_unique_index(nl_hi,nl_wi,hi,wi,
                     wsOff_h,wsOff_w,time_offset,
                     stride0,stride1,ws,wsHalf,full_ws):

    # -- check spatial coordinates --
    num_h = (nl_hi - hi)#//stride1
    num_w = (nl_wi - wi)#//stride1
    # num_h = (nl_hi - hi)//stride0
    # num_w = (nl_wi - wi)//stride0
    vprint("num_h,num_w,wsHalf: ",num_h,num_w,wsHalf)

    # -- check oob --
    oob_i = abs(num_h) > wsHalf
    oob_j = abs(num_w) > wsHalf

    # -- oob names --
    if oob_i and oob_j:
        vprint("case0")
        # -- check offset --
        adj_h = wsHalf - wsOff_h
        adj_w = wsHalf - wsOff_w
        vprint("adj_h,adj_w: ",adj_h,adj_w)

        # -- di,dj --
        # di = wsHalf - abs(adj_h)
        # dj = wsHalf - abs(adj_w)
        di = wsHalf - adj_h if adj_h > 0 else adj_h
        dj = wsHalf - adj_w if adj_w > 0 else adj_w

        # -- small square --
        # mi = di + wsHalf*dj
        # ws_i = mi % ws
        # ws_j = mi // ws + (ws-1)
        mi = di + wsHalf*dj
        ws_i = mi % ws
        ws_j = mi // ws + (ws-1)
        vprint("[case0] ws_i,ws_j: ",ws_i,ws_j)

        ws_i = (abs(adj_h)-1)//stride0
        ws_j = (abs(adj_w)-1)//stride0

        # wsNum = (ws-1)//stride0+1

        # # -- check offset --
        # adj_h = wsHalf - wsOff_h
        # adj_w = wsHalf - wsOff_w
        # adj_h = adj_h//stride0
        # adj_w = adj_w//stride0

        # # -- di,dj --
        # di = wsHalf - abs(adj_h)
        # dj = wsHalf - abs(adj_w)
        # di = di//stride0
        # dj = dj//stride0

        # # -- small square --
        # mi = di + wsHalf*dj
        # ws_i = mi % wsNum
        # ws_j = mi // wsNum + (wsNum-1)

    elif oob_i and not(oob_j):
        vprint("case1")
        ws_j = abs(num_h) - (wsHalf+1)
        ws_i = num_w+wsHalf
    elif oob_j and not(oob_i):
        vprint("case2")
        # ws_j = abs(num_w) - (wsHalf+1) + (wsHalf)
        # ws_i = num_h+wsHalf
        ws_j = abs(num_w) - (wsHalf+1)
        ws_i = num_h+wsHalf

    # -- debug --
    # print((hi,wi),(nl_hi,nl_wi),(oob_i,oob_j,))

    # -- standard names --
    if not(oob_i or oob_j):
        ws_i = num_h + wsHalf
        ws_j = num_w + wsHalf
        # print(num_h,num_w,wsHalf,ws_i,ws_j)

    # -- check oob --
    wsNum = (ws-1)//stride0+1
    xor_oob = (oob_i or oob_j) and not(oob_i and oob_j) and full_ws
    and_oob = (oob_i and oob_j) and full_ws

    # -- divide out stride0 --
    vprint("pre: ",ws_i,ws_j)
    # if not(oob_i or oob_j):
    if not(and_oob):
        ws_i = ws_i//stride0
        ws_j = ws_j//stride0
    vprint("post: ",ws_i,ws_j)

    # -- check --
    if not(oob_i and oob_j):
        assert (ws_i >= 0) and (ws_i < ws)
        assert (ws_j >= 0) and (ws_j < ws)

    # -- get unique index --
    # wsNum = (ws-1)//stride0+1
    # li = (ws_i) + (ws_j)*wsNum + time_offset
    # # li = li + wsNum*wsNum if oob else li
    # xor_oob = (oob_i or oob_j) and not(oob_i and oob_j)
    # li = li + wsNum*wsNum if xor_oob else li
    # print(li)
    # # 2*(wsNum//2)*wsNum
    # li = li + wsNum*wsNum+2*(wsNum//2)*wsNum if (oob_i and oob_j) else li
    # print(li)

    # -- get unique index --
    if not(oob_i or oob_j):
        li = (ws_i) + (ws_j)*wsNum + time_offset
    # elif xor_oob:
    #     li = (ws_i) + (ws_j)*wsNum + time_offset + wsNum*wsNum
    elif xor_oob and oob_i:
        li = (ws_i) + (ws_j)*wsNum + time_offset + wsNum*wsNum
    elif xor_oob and oob_j:
        li = (ws_i) + (ws_j)*wsNum + (wsNum//2)*wsNum + time_offset + wsNum*wsNum
    elif (and_oob):
        # ws_i = abs(ws_i-1)//stride0-1
        # ws_j = abs(ws_j-1)//stride0-1
        # ws_i = ws_i
        # ws_j = ws_j % stride0
        vprint("[case0] ws_i,ws_j: ",ws_i,ws_j)
        li = (ws_i) + (ws_j)*(wsNum//2)
        li = li + time_offset + wsNum*wsNum + 2*(wsNum//2)*wsNum
        # li = 15
    else:
        raise ValueError("What?")

    # # li = li + wsNum*wsNum if oob else li
    # li = li + wsNum*wsNum if xor_oob else li
    # print(li)
    # # 2*(wsNum//2)*wsNum
    # li = li + wsNum*wsNum+2*(wsNum//2)*wsNum if (oob_i and oob_j) else li
    # print(li)


    return li,and_oob


def get_tlims(ti, T, wt):
  t_shift = min(0,ti - wt) + max(0,ti + wt - (T-1));
  t_min = max(ti - wt - t_shift,0);
  t_max = min(T-1,ti + wt - t_shift);
  return t_max,t_min

def fill_names(ti,h_ref,w_ref,ki,ws,wt,stride0,stride1,st_offset,
               full_ws,names,counts,flows_k):

    # -- unpack --
    W_t = 2*wt+1
    S,T,H,W,_ = names.shape
    wsHalf = (ws-1)//2
    wsOff_h,wsOff_w = wsHalf,wsHalf

    # -- get non-local position --
    hi,wi = h_ref*stride0,w_ref*stride0
    nl_ti = ti + flows_k[ti][h_ref][w_ref][ki][0]
    nl_hi = hi + flows_k[ti][h_ref][w_ref][ki][1]
    nl_wi = wi + flows_k[ti][h_ref][w_ref][ki][2]
    valid = check_valid(nl_ti,nl_hi,nl_wi,T,H,W)
    if not(valid): return
    # if (wi < 8) or (hi < 8): return
    # if (wi > 56) or (hi > 56): return
    # if not((nl_hi == 16) and (nl_wi == 9)): return
    # if not((nl_hi == 4) and (nl_wi == 16)): return
    # if not((wi == 0) and (hi == 0)): return
    # if not((nl_hi == 1) and (nl_wi == 0)): return
    # if not((nl_hi == 0) and (nl_wi == 2)): return
    # if not((nl_hi == 0) and (nl_wi == 2)): return
    # if not((nl_hi == 2) and (nl_wi == 2)): return
    # if not((nl_hi == 0) and (nl_wi == 3)): return
    # if not((nl_hi == 4) and (nl_wi == 4)): return
    # if not((nl_hi == 0) and (nl_wi == 5)): return
    # if not((nl_hi == 6) and (nl_wi == 6)): return

    # -- search flow from difference --
    t_max,t_min = get_tlims(ti, T, wt)
    dt = nl_ti - ti
    dto = t_max - ti
    si = (dt-st_offset) if (dt >= 0) else (dto - dt - st_offset)
    # ws_ti = (ti+nl_ti) % W_t
    ws_ti = (nl_ti+ti) % T if W_t > 1 else 0
    # print(si,dt,st_offset,nl_ti,ti)

    # -- offset search offsets --
    wsOff_h,wsOff_w = set_search_offsets(wsOff_h, wsOff_w, hi, wi,
                                         stride1, wsHalf, ws, H, W, full_ws)

    # -- get search index --
    # ws_i = (nl_hi - hi)//stride1 + wsOff_h
    # ws_j = (nl_wi - wi)//stride1 + wsOff_w
    # ws_i_orig = ws_i
    # ws_j_orig = ws_j

    # -- handle oob --
    time_offset = ws_ti*(ws*ws+2*(ws//2)*ws+(ws//2)**2)
    li,oob = get_unique_index(nl_hi,nl_wi,hi,wi,
                              wsOff_h,wsOff_w,time_offset,
                              stride0,stride1,ws,wsHalf,full_ws)
    # nl_hi,nl_wi,hi,wi,stride1,ws,
    # wsHalf,wsOff_h,wsOff_w,time_offset,full_ws)
    # ws_i,ws_j,oob = check_oob(ws_i,ws_j,nl_hi,nl_wi,hi,wi,stride1,ws,
    #                           wsHalf,wsOff_h,wsOff_w,full_ws)
    # if not(oob): return

    print("Ref/NonLocal: ",(ti,hi,wi),(nl_ti,nl_hi,nl_wi),li,ws_ti,dt,oob,stride1,ws)

    # -- update --
    # assert((ws_ti >= 0) and (ws_ti <= (W_t-1)))
    wsNum = (ws-1)//stride0+1
    if np.any(names[li,ti,hi,wi]<0):
        names[li,ti,hi,wi,...] = 0
    names[li,nl_ti,nl_hi,nl_wi,0] = ti
    names[li,nl_ti,nl_hi,nl_wi,1] = hi
    names[li,nl_ti,nl_hi,nl_wi,2] = wi
    if counts[li,nl_ti,nl_hi,nl_wi] == 0:
        print("already here.")
        exit()
    if li < wsNum*wsNum and oob:
        print("li is small to be out of bounds.")
        exit()
    counts[li,nl_ti,nl_hi,nl_wi] += 1

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    # ws = 7
    wt = 0
    W_t = 2*wt+1
    full_ws = False
    T,H,W = 1,64,64
    stride0 = 8
    ws = 2*stride0 + 1
    # ws = 3
    # ws = 17
    # assert( ws>=(2*stride0+1) )
    stride1 = 1
    W_t_num = T if wt > 0 else 1#min(W_t + 2*wt,T)
    wsNum = (ws-1)//stride0+1
    # wsNum = (ws//2)//stride0+1
    print(wsNum)
    #S = W_t_num*(ws*ws + 2*(ws//2)*ws + (ws//2)**2)
    # S = W_t_num*(wsNum*wsNum + 2*(wsNum//2)*wsNum+(wsNum//2)**2)
    S = W_t_num*(wsNum*wsNum + 2*(wsNum//2)*wsNum+(wsNum//2)**2)
    print("wsNum,wsNum*wsNum,2*(wsNum//2)*wsNum: ",wsNum,wsNum*wsNum,2*(wsNum//2)*wsNum)
    print("S: ",S)
    # S = 17
    # S = 16
    print("ws,wt,S: ",ws,wt,S)
    vals = np.zeros((T,H,W,ws,ws))
    names = -np.ones((S,T,H,W,3))
    counts = -np.ones((S,T,H,W))
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    ws_og = ws
    ws = ws_og + (H-(nH-1)*stride0)
    flows_k = get_data(ws,ws_og,wt,T,H,W,stride0,stride1,full_ws)
    K = flows_k.shape[-2]
    st_offset = 1
    print("flows_k.shape: ",flows_k.shape)
    for ti in range(T):
        for h_ref in range(nH):
            for w_ref in range(nW):
                for ki in range(K):
                    fill_names(ti,h_ref,w_ref,ki,ws,wt,stride0,stride1,
                               st_offset,full_ws,names,counts,flows_k)

    # # graph_transpose_q2k(dists_k,flows_k,flows,ws,wt,stride0,H,W,full_ws):
    # scatter_flows_k = stnls.graph_opts.scatter_tensor(flows_k,flows_k,labels,
    #                                                   stride0,stride1,H,W)


    # print(counts[:,0,2,2])
    # print(counts[:,0,:3,:3].T)
    # print(counts[:,0,59,3])
    print(names.shape,counts.shape)
    print(names[:,0,-2:,-2:].squeeze())
    print(counts[:,0,-2:,-2:].squeeze())
    print(counts[:,0,16,9])
    # for i in range(S):
    #     print(counts[i,0])
    print(np.sum(counts>=0,0).max())
    print(np.sum(counts>=0),T*nH*nW*K)
    print(np.sum(counts==0),T*nH*nW*K)
    print("wsNum,wsNum*wsNum,2*(wsNum//2)*wsNum: ",wsNum,wsNum*wsNum,2*(wsNum//2)*wsNum)
    print("S: ",S)


if __name__ == "__main__":
    main()
