

import torch as th

def get_time_window_inds(ti,wt,T):
    swap = False
    t_inc = 0
    prev_t = ti
    t_shift = min(0,ti-wt) + max(0,ti + wt - (T-1))
    t_max = min(T-1,ti + wt - t_shift);
    # print(t_shift,t_max)
    tj = ti
    inds = []
    for _tj in range(2*wt+1):
        # -- update search frame --
        prev_t = tj
        tj = prev_t + t_inc
        swap = tj > t_max
        t_inc = 1 if (t_inc == 0) else t_inc
        t_inc = -1 if swap else t_inc
        tj = ti-1 if swap else tj
        prev_t = ti if swap else prev_t
        # print(ti,tj,t_inc,swap)
        inds.append(tj)
    return inds

def paired_vids(forward, vid0, vid1, acc_flows, wt, skip_self=False):
    dists,inds = [],[]
    T = vid0.shape[1]
    zflow = th.zeros_like(acc_flows.fflow[:,0,0])
    for ti in range(T):
        # if ti != 1: continue

        swap = False
        t_inc = 0
        prev_t = ti
        t_shift = min(0,ti-wt) + max(0,ti + wt - (T-1))
        t_max = min(T-1,ti + wt - t_shift);
        # print(t_shift,t_max)
        tj = ti

        dists_i,inds_i = [],[]
        for _tj in range(2*wt+1):

            # -- update search frame --
            prev_t = tj
            tj = prev_t + t_inc
            swap = tj > t_max
            t_inc = 1 if (t_inc == 0) else t_inc
            t_inc = -1 if swap else t_inc
            tj = ti-1 if swap else tj
            prev_t = ti if swap else prev_t
            # print(ti,tj,t_inc,swap)

            frame0 = vid0[:,ti]
            frame1 = vid1[:,tj]
            if (ti == tj) and skip_self: continue
            if ti == tj:
                flow = zflow
            elif ti < tj:
                # print("fwd: ",ti,tj,tj-ti-1)
                # flow = acc_flows.fflow[:,tj - ti - 1]
                flow = acc_flows.fflow[:,ti,tj-ti-1]
            elif ti > tj:
                # print("bwd: ",ti,tj,ti-tj-1)
                # flow = acc_flows.bflow[:,ti - tj - 1]
                flow = acc_flows.bflow[:,ti,ti-tj-1]
            flow = flow.float()
            dists_ij,inds_ij = forward(frame0,frame1,flow)
            inds_t = tj*th.ones_like(inds_ij[...,[0]])
            inds_ij = th.cat([inds_t,inds_ij],-1)
            # print("inds_ij.shape: ",inds_ij.shape,inds_t.shape)
            dists_i.append(dists_ij)
            inds_i.append(inds_ij)
        dists_i = th.cat(dists_i,-1)
        inds_i = th.cat(inds_i,-2)
        dists.append(dists_i)
        inds.append(inds_i)
    dists = th.cat(dists,-2)
    inds = th.cat(inds,-3)
    # print("inds.shape: ",inds.shape)
    return dists,inds
