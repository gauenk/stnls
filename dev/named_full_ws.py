

import math
import stnls
import torch as th
import numpy as np
import random


def get_data(ws,wt,T,H,W,stride0,stride1,full_ws):

    # -- get args --
    seed = 123
    nheads = 1
    device = "cuda:0"
    set_seed(seed)
    W_t = 2*wt+1
    B,HD,F = 1,1,1
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    K = ws*ws*W_t
    itype = "int"

    # -- load flows --
    flows = th.ones((B,HD,T,W_t-1,2,nH,nW)).cuda()/2.
    flows = th.rand_like(flows)/2.+th.randint_like(flows,-3,3)+0.2
    flows = th.zeros_like(flows)
    flows = flows.round().int()

    # -- load top-k flows --
    ps = 3
    search = stnls.search.NonLocalSearch(ws, wt, ps, K, nheads, dist_type="l2",
                                         stride0=stride0, stride1=stride1,
                                         reflect_bounds=True,self_action="anchor",
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

def check_oob(ws_i,ws_j,nl_hi,nl_wi,hi,wi,
              stride1,ws,wsHalf,wsOff_h,wsOff_w,full_ws):

    # -- check spatial coordinates --
    ws_i_tmp = (nl_hi - hi)//stride1# + wsHalf
    ws_j_tmp = (nl_wi - wi)//stride1# + wsHalf
    # assert(ws_i_tmp >= -wsHalf);
    # assert(ws_j_tmp >= -wsHalf);
    # assert(ws_i_tmp < ws+wsHalf);
    # assert(ws_j_tmp < ws+wsHalf);

    # -- max search --
    hMax = hi + stride1 * ((ws-1) - wsHalf)
    wMax = wi + stride1 * ((ws-1) - wsHalf)
    hMin = (hi - stride1 * wsHalf)
    wMin = (wi - stride1 * wsHalf)

    # -- check offset --
    delta_h = wsHalf - wsOff_h
    delta_w = wsHalf - wsOff_w

    # int delta_i,delta_j;
    delta_i = (ws-1-ws_i) if (delta_h > 0) else (ws_i if (delta_h<0) else ws)
    oob_i = (delta_h != 0) and (abs(delta_h) >= delta_i)
    delta_j = (ws-1-ws_j) if (delta_w > 0) else (ws_j if (delta_w<0) else ws)
    oob_j = (delta_w != 0) and (abs(delta_w) >= delta_j)
    print("oob_i,oob_j,(ws_i,ws_j),(ws_i_tmp,ws_j_tmp): ",
          oob_i,oob_j,(ws_i,ws_j),(ws_i_tmp,ws_j_tmp),(delta_i,delta_j))
    ws_i_noadj = ws_i_tmp + wsHalf
    ws_j_noadj = ws_j_tmp + wsHalf
    if oob_i and oob_j:
        ws_j = (ws-1)//2+1
        ws_i = 0
    elif oob_i and not(oob_j):
        # ws_i = ws_j
        ws_i = ws-1-ws_j_noadj
        ws_j = 0
        # ws_i = ws_i + ws
        # if ws_i_tmp < 0:
        #     ws_i = abs((ws_i_tmp - ws))
        # else:
        #     ws_i = abs((ws_i_tmp - ws))
    elif oob_j and not(oob_i):
        print("oob_j: ",oob_j,ws_i_noadj,ws_j_noadj)
        ti = ws_i_noadj + ws_j_noadj * ws
        print(ti)
        ws_j = 1#ws_j
        ws_i = ws-1-ws_i_noadj
    print("(ws_i,ws_j): ",(ws_i,ws_j))
    if wsHalf != wsOff_h and not(oob_i or oob_j):
        ws_i = ws_i_tmp+wsHalf
    if wsHalf != wsOff_w and not(oob_i or oob_j):
        ws_j = ws_j_tmp+wsHalf

    # if wsHalf == wsOff_h and not(oob_i or oob_j):
    #     ws_i = ws_i_tmp+wsHalf
    # if wsHalf == wsOff_w and not(oob_i or oob_j):
    #     ws_j = ws_j_tmp+wsHalf


        # if ws_j_tmp < 0:
        #     ws_j = abs(ws_j_tmp - ws)
        # else:
        #     ws_j = abs(ws_j_tmp - ws)
    # ws_i = (ws_i_tmp - ws) % ws if oob_i  else ws_i_tmp
    # ws_j = (ws_j_tmp - ws) % ws if oob_j else ws_j_tmp

    # -- new idea --
    dH = abs(nl_hi - hi)
    dW = abs(nl_wi - wi)

    # -- check oob --
    oob = (oob_i or oob_j) and full_ws
    # ws_i = delta_i if oob else ws_i_tmp
    # ws_j = delta_j if oob else ws_j_tmp
    # # ws_i = ws_i_tmp % ws# if oob else ws_i
    # # ws_j = ws_j_tmp % ws# if oob else ws_j
    gt_i = ws_i_tmp >= ws
    gt_j = ws_j_tmp >= ws

    # if gt_i:
    #     ws_i = 2*(hMax-1-hi) - ws_i
    # if gt_j:
    #     ws_j = 2*(wMax-1-wi) - ws_j

    print((ws_i,ws_j),(hMax,wMax))
    return ws_i,ws_j,oob,gt_i,gt_j


def fill_names(ti,hi,wi,ki,ws,wt,stride0,stride1,full_ws,names,counts,flows_k):

    # -- unpack --
    S,T,H,W,_ = names.shape
    wsHalf = (ws-1)//2
    wsOff_h,wsOff_w = wsHalf,wsHalf

    # -- get non-local position --
    nl_ti = ti + flows_k[ti][hi][wi][ki][0]
    nl_hi = hi + flows_k[ti][hi][wi][ki][1]
    nl_wi = wi + flows_k[ti][hi][wi][ki][2]
    valid = check_valid(nl_ti,nl_hi,nl_wi,T,H,W)
    if not(valid): return
    # if not((wi == 0) or (hi == 0)): return
    # if not((nl_hi == 1) and (nl_wi == 0)): return
    # if not((nl_hi == 0) and (nl_wi == 2)): return
    # if not((nl_hi == 2) and (nl_wi == 2)): return


    # -- offset search offsets --
    wsOff_h,wsOff_w = set_search_offsets(wsOff_h, wsOff_w, hi, wi,
                                         stride1, wsHalf, ws, H, W, full_ws)

    # -- get search index --
    ws_i = (nl_hi - hi)//stride1 + wsOff_h
    ws_j = (nl_wi - wi)//stride1 + wsOff_w
    ws_i_orig = ws_i
    ws_j_orig = ws_j

    # -- handle oob --
    ws_i,ws_j,oob,gt_i,gt_j = check_oob(ws_i,ws_j,nl_hi,nl_wi,hi,wi,stride1,ws,
                                        wsHalf,wsOff_h,wsOff_w,full_ws)
    # if not(oob): return

    # -- get unique index --
    li = (ws_i) + (ws_j)*ws
    li_off = ws-1 if (ws_i_orig < ws_j_orig) else 0
    print(gt_i,gt_j)
    li_off = 0
    # li = li + ws*ws + li_off if oob else li
    li = li + ws*ws + li_off if oob else li
    print("Ref/NonLocal: ",(hi,wi),(nl_hi,nl_wi),li,oob)

    # -- update --
    if np.any(names[li,ti,hi,wi]<0):
        names[li,ti,hi,wi,...] = 0
    names[li,nl_ti,nl_hi,nl_wi,0] = ti
    names[li,nl_ti,nl_hi,nl_wi,1] = hi
    names[li,nl_ti,nl_hi,nl_wi,2] = wi
    if counts[li,nl_ti,nl_hi,nl_wi] == 0:
        exit()
    if li < ws*ws and oob:
        exit()
    counts[li,nl_ti,nl_hi,nl_wi] += 1

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    ws = 3
    wt = 0
    full_ws = True
    T,H,W = 1,8,8
    stride0,stride1 = 1,1
    S = 2*ws*ws
    vals = np.zeros((T,H,W,ws,ws))
    names = -np.ones((S,T,H,W,3))
    counts = -np.ones((S,T,H,W))
    flows_k = get_data(ws,wt,T,H,W,stride0,stride1,full_ws)
    K = flows_k.shape[-2]
    print("flows_k.shape: ",flows_k.shape)
    for ti in range(T):
        for hi in range(H):
            for wi in range(W):
                for ki in range(K):
                    fill_names(ti,hi,wi,ki,ws,wt,stride0,stride1,
                               full_ws,names,counts,flows_k)

    print(counts[:,0,2,2])
    print(counts[:,0,:3,:3].T)
    # for i in range(S):
    #     print(counts[i,0])
    print(np.sum(counts>=0),T*H*W*K)
if __name__ == "__main__":
    main()
