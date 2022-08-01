
# -- python-only kernel --
from numba import cuda,jit
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat


def run(vid0,iqueries,flow,k,ps,pt,ws,wt,chnls,dilation=1,stride=1,
        use_adj=True,reflect_bounds=True,search_abs=False,
        h0_off=0,w0_off=0,h1_off=0,w1_off=0,vid1=None):

    # -- select alt vid --
    if vid1 is None: vid1 = vid0

    # -- allocate --
    use_k = k > 0
    k,ws = get_args(k,ws,wt,stride,vid0.shape)
    device = iqueries.device
    nq = iqueries.shape[0]
    dists_exh,inds_exh = allocate_exh(nq,ws,wt,device)

    # -- unpack --
    fflow,bflow = unpack_flow(flow,vid0.shape,device)

    # -- exec --
    numba_search_launcher(vid0,vid1,iqueries,dists_exh,inds_exh,fflow,bflow,
                          h0_off,w0_off,h1_off,w1_off,
                          k,ps,pt,ws,wt,chnls,dilation,stride,
                          use_adj,reflect_bounds,search_abs)

    # -- patches of topk --
    if use_k:
        dists,inds = allocate_k(nq,k,device)
        get_topk(dists_exh,inds_exh,dists,inds)
        # dists[:,0] = 0. # fix the "-100" hack to 0.
    else:
        b = dists_exh.shape[0]
        # args = th.where(dists_exh < 0)
        # dists_exh[args] = 0.
        dists = dists_exh.view(b,-1)
        inds = inds_exh.view(b,-1,3)

    return dists,inds

def get_args(k,ws,wt,stride,vshape):
    t,c,h,w = vshape
    if ws <= 0: ws = int((h-1)//stride+1)
    if k <= 0: k = ws * ws * (2*(wt+1))
    return k,ws

def unpack_flow(flow,shape,device):
    t,c,h,w = shape
    if flow is None:
        zflow = th.zeros((t,2,h,w),device=device,dtype=th.float32)
        fflow = zflow
        bflow = zflow
    else:
        fflow = flow.fflow
        bflow = flow.bflow
    return fflow,bflow

def allocate_k(nq,k,device):
    dists = th.zeros((nq,k),device=device,dtype=th.float32)
    dists[...] = float("inf")
    inds = th.zeros((nq,k,3),device=device,dtype=th.int32)
    inds[...] = -1
    return dists,inds

def allocate_exh(nq,ws,wt,device):
    dists = th.zeros((nq,2*wt+1,ws,ws),device=device,dtype=th.float32)
    dists[...] = float("inf")
    inds = th.zeros((nq,2*wt+1,ws,ws,3),device=device,dtype=th.int32)
    inds[...] = -1
    return dists,inds

def get_topk(l2_vals,l2_inds,vals,inds):

    # -- reshape exh --
    nq,st,ws,ws = l2_vals.shape
    l2_vals = l2_vals.view(nq,-1)
    l2_inds = l2_inds.view(nq,-1,3)

    # -- shape info --
    b,_ = l2_vals.shape
    _,k = vals.shape

    # -- take mins --
    order = th.argsort(l2_vals,dim=1,descending=False)
    vals[:b,:] = th.gather(l2_vals,1,order[:,:k])
    for i in range(inds.shape[-1]):
        inds[:b,:,i] = th.gather(l2_inds[:,:,i],1,order[:,:k])

#
# -- Numba --
#

def create_frame_range(nframes,nWt_f,nWt_b,ps_t,device):
    tranges,n_tranges,min_tranges = [],[],[]
    for t_c in range(nframes-ps_t+1):

        # -- limits --
        shift_t = min(0,t_c - nWt_b) + max(0,t_c + nWt_f - nframes + ps_t)
        t_start = max(t_c - nWt_b - shift_t,0)
        t_end = min(nframes - ps_t, t_c + nWt_f - shift_t)+1

        # -- final range --
        trange = [t_c]
        trange_s = np.arange(t_c+1,t_end)
        trange_e = np.arange(t_start,t_c)[::-1]
        for t_i in range(trange_s.shape[0]):
            trange.append(trange_s[t_i])
        for t_i in range(trange_e.shape[0]):
            trange.append(trange_e[t_i])

        # -- aug vars --
        n_tranges.append(len(trange))
        min_tranges.append(np.min(trange))

        # -- add padding --
        for pad in range(nframes-len(trange)):
            trange.append(-1)

        # -- to tensor --
        trange = th.IntTensor(trange).to(device)
        tranges.append(trange)

    tranges = th.stack(tranges).to(device)
    n_tranges = th.IntTensor(n_tranges).to(device)
    min_tranges = th.IntTensor(min_tranges).to(device)
    return tranges,n_tranges,min_tranges

def numba_search_launcher(vid0,vid1,iqueries,dists,inds,
                          fflow,bflow,
                          h0_off,w0_off,h1_off,w1_off,
                          k,ps,pt,ws,wt,chnls,
                          dilation,stride,
                          use_adj,reflect_bounds,search_abs):

    # -- buffer for searching --
    t = vid0.shape[0]
    nq = inds.shape[0]
    device = inds.device
    bufs = th.zeros(nq,3,t,ws,ws,dtype=th.int32,device=device)

    # -- pre-computed search offsets --
    tranges,n_tranges,min_tranges = create_frame_range(t,wt,wt,pt,device)

    # -- numbify all params --
    vid0_nba = cuda.as_cuda_array(vid0)
    vid1_nba = cuda.as_cuda_array(vid1)
    iqueries_nba = cuda.as_cuda_array(iqueries)
    dists_nba = cuda.as_cuda_array(dists)
    inds_nba = cuda.as_cuda_array(inds)
    fflow_nba = cuda.as_cuda_array(fflow)
    bflow_nba = cuda.as_cuda_array(bflow)
    bufs_nba = cuda.as_cuda_array(bufs)
    tranges_nba = cuda.as_cuda_array(tranges)
    n_tranges_nba = cuda.as_cuda_array(n_tranges)
    min_tranges_nba = cuda.as_cuda_array(min_tranges)
    # cs_nba = cuda.external_stream(cs)

    # -- launch params --
    nq = iqueries.shape[0]
    batches_per_block = 10
    bpb = batches_per_block

    # -- launch params --
    w_threads = min(ws,32)
    nthreads = (w_threads,w_threads)
    ws_iters = (ws-1)//w_threads + 1
    nblocks = (nq-1)//batches_per_block+1

    # -- exec kernel --
    numba_search[nblocks,nthreads](vid0_nba,vid1_nba,iqueries_nba,
                                   dists_nba,inds_nba,
                                   fflow_nba,bflow_nba,
                                   h0_off,w0_off,h1_off,w1_off,
                                   ps,pt,chnls,dilation,stride,
                                   bufs_nba,tranges_nba,n_tranges_nba,
                                   min_tranges_nba,
                                   use_adj,reflect_bounds,search_abs,
                                   ws_iters,bpb)


# @cuda.jit(debug=True,max_registers=64,opt=False)
@cuda.jit(debug=False,max_registers=64)
def numba_search(vid0,vid1,iqueries,dists,inds,fflow,bflow,
                 h0_off,w0_off,h1_off,w1_off,
                 ps,pt,chnls,dilation,stride,
                 bufs,tranges,n_tranges,min_tranges,
                 use_adj,reflect_bounds,search_abs,ws_iters,bpb):

    # -- reflective boundary --
    def bounds(val,lim):
        nval = val
        if val < 0: nval = -nval
        elif val >= lim: nval = 2*(lim-1)-nval
        return int(nval)

    # -- shapes --
    nframes,color,h,w = vid0.shape
    height,width = h,w
    bsize,st,ws,ws = dists.shape
    bsize,st,ws,ws,_ = inds.shape
    Z = ps*ps*pt*chnls
    psHalf = int((ps)//2)
    wsHalf = (ws)//2
    adj = psHalf if use_adj else 0

    # -- cuda threads --
    cu_tidX = cuda.threadIdx.x
    cu_tidY = cuda.threadIdx.y
    blkDimX = cuda.blockDim.x
    blkDimY = cuda.blockDim.y
    # tidX = cuda.threadIdx.x
    # tidY = cuda.threadIdx.y

    # ---------------------------
    #
    #      search frames
    #
    # ---------------------------

    # -- access with blocks and threads --
    block_start = cuda.blockIdx.x*bpb

    # -- we want enough work per thread, so we process multiple per block --
    for _bidx in range(bpb):

        # ---------------------------
        #    extract anchor pixel
        # ---------------------------

        bidx = block_start + _bidx
        if bidx >= iqueries.shape[0]: continue

        # -- unpack pixel locs --
        ti = iqueries[bidx,0]
        hi = iqueries[bidx,1]
        wi = iqueries[bidx,2]

        # ---------------------------
        #     valid (anchor pixel)
        # ---------------------------

        valid_t = (ti < nframes) and (ti >= 0)
        valid_top = (hi < height) and (hi >= 0)
        valid_left = (wi < width) and (wi >= 0)
        valid_anchor = valid_t and valid_top and valid_left

        # ---------------------------------------
        #     searching loop for (ti,top,left)
        # ---------------------------------------

        trange = tranges[ti]
        n_trange = n_tranges[ti]
        min_trange = min_tranges[ti]

        # -- we loop over search space if needed --
        for _xi in range(ws_iters):
            tidX = cu_tidX + blkDimX*_xi
            ws_i = tidX
            if ws_i >= ws: continue

            for _yi in range(ws_iters):
                tidY = cu_tidY + blkDimY*_yi
                ws_j = tidY
                if ws_j >= ws: continue

                for tidZ in range(n_trange):
                    wt_k = tidZ

                    # -------------------
                    #    search frame
                    # -------------------
                    n_ti = trange[wt_k]
                    dt = trange[wt_k] - min_trange

                    # ------------------------
                    #      init direction
                    # ------------------------

                    direction = max(-1,min(1,n_ti - ti))
                    if direction != 0:

                        # -- get offset at index --
                        dtd = int(dt-direction)
                        cw0 = bufs[bidx,0,dtd,ws_i,ws_j]
                        ch0 = bufs[bidx,1,dtd,ws_i,ws_j]
                        ct0 = bufs[bidx,2,dtd,ws_i,ws_j]

                        # -- legalize access --
                        l_cw0 = int(max(0,min(w-1,cw0)))
                        l_ch0 = int(max(0,min(h-1,ch0)))
                        l_ct0 = int(max(0,min(nframes-1,ct0)))

                        # -- pick flow --
                        flow = fflow if direction > 0 else bflow

                        # -- access flows --
                        cw_f = cw0 + flow[l_ct0,0,l_ch0,l_cw0]
                        ch_f = ch0 + flow[l_ct0,1,l_ch0,l_cw0]

                        # -- round --
                        cw_f = int(cw_f + 0.5)
                        ch_f = int(ch_f + 0.5)

                        # -- bounds --
                        cw = max(0,min(width-1,cw_f))
                        ch = max(0,min(height-1,ch_f))
                        ct = n_ti

                    else:
                        cw = wi
                        ch = hi
                        ct = ti


                    # ----------------
                    #     update
                    # ----------------
                    bufs[bidx,0,dt,tidX,tidY] = cw
                    bufs[bidx,1,dt,tidX,tidY] = ch
                    bufs[bidx,2,dt,tidX,tidY] = ct
                    # cw = wi
                    # ch = hi
                    # ct = n_ti

                    # --------------------
                    #      init dists
                    # --------------------
                    dist = 0

                    # -----------------
                    #    spatial dir
                    # -----------------
                    if search_abs:
                        n_hi = stride * ws_i
                        n_wi = stride * ws_j
                    else:
                        n_hi = ch + stride * (ws_i - wsHalf)
                        n_wi = cw + stride * (ws_j - wsHalf)

                    # ---------------------------
                    #      valid (search "n")
                    # ---------------------------
                    valid_n_t = (n_ti < nframes) and (n_ti >= 0)
                    valid_n_top = (n_hi < height) and (n_hi >= 0)
                    valid_n_left = (n_wi < width) and (n_wi >= 0)
                    valid_prop = valid_n_t and valid_n_top and valid_n_left
                    valid = valid_prop and valid_anchor

                    # ---------------------------------
                    #
                    #  compute delta over patch vol.
                    #
                    # ---------------------------------

                    for pk in range(pt):
                        for pi in range(ps):
                            for pj in range(ps):

                                # -- inside entire image --
                                vH = (hi-h0_off) + dilation*(pi - psHalf + adj)
                                vW = (wi-w0_off) + dilation*(pj - psHalf + adj)
                                vH = bounds(vH,height) if reflect_bounds else vH
                                vW = bounds(vW,width) if reflect_bounds else vW
                                vT = ti + pk

                                nH = (n_hi-h1_off) + dilation*(pi - psHalf + adj)
                                nW = (n_wi-w1_off) + dilation*(pj - psHalf + adj)
                                nH = bounds(nH,height) if reflect_bounds else nH
                                nW = bounds(nW,width) if reflect_bounds else nW
                                nT = n_ti + pk

                                # -- valid checks [for testing w/ zero pads] --
                                vvalid = (vH < height) and (vH >= 0)
                                vvalid = vvalid and (vW < width) and (vW >= 0)
                                vvalid = vvalid and (vT < nframes) and (vT >= 0)

                                nvalid = (nH < height) and (nH >= 0)
                                nvalid = nvalid and (nW < width) and (nW >= 0)
                                nvalid = nvalid and (nT < nframes) and (nT >= 0)

                                # -- all channels --
                                for ci in range(chnls):

                                    # -- get data --
                                    if vvalid:
                                        v_pix = vid0[vT][ci][vH][vW]
                                    else:
                                        v_pix = 0.

                                    if nvalid:
                                        n_pix = vid1[nT][ci][nH][nW]
                                    else:
                                        n_pix = 0.

                                    # -- compute dist --
                                    if valid:
                                        dist += (v_pix - n_pix)**2

                    # -- dists --
                    if not(valid): dist = np.inf
                    dists[bidx,wt_k,ws_i,ws_j] = dist

                    # -- inds --
                    inds[bidx,wt_k,ws_i,ws_j,0] = n_ti
                    inds[bidx,wt_k,ws_i,ws_j,1] = n_hi
                    inds[bidx,wt_k,ws_i,ws_j,2] = n_wi

                    # -- final check [put self@index 0] --
                    # eq_ti = n_ti == ti
                    # eq_hi = n_hi == hi # hi
                    # eq_wi = n_wi == wi # wi
                    # eq_dim = eq_ti and eq_hi and eq_wi
                    # dist = dists[bidx,wt_k,ws_i,ws_j]
                    # dists[bidx,wt_k,ws_i,ws_j] = -100 if eq_dim else dist



