
# -- python-only kernel --
from numba import cuda,jit

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

def run(vid,queryInds,flow,k,ps,pt,ws,wt,chnls,dilation=1,stride=1):

    # -- allocate --
    device = queryInds.device
    nq = queryInds.shape[0]
    nlDists,nlInds = allocate_k(nq,k,device)
    nlDists_exh,nlInds_exh = allocate_exh(nq,ws,wt,device)

    # -- unpack --
    fflow,bflow = unpack_flow(flow,vid.shape,device)

    # -- exec --
    numba_search_launcher(vid,queryInds,nlDists_exh,nlInds_exh,
                          fflow,bflow,k,ps,pt,ws,wt,chnls,dilation,stride)

    # -- patches of topk --
    get_topk(nlDists_exh,nlInds_exh,nlDists,nlInds)
    nlDists[:,0] = 0. # fix the "-100" hack to 0.

    return nlDists,nlInds

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

def numba_search_launcher(vid,queryInds,nlDists,nlInds,
                          fflow,bflow,k,ps,pt,ws,wt,chnls,dilation,stride):

    # -- buffer for searching --
    t = vid.shape[0]
    nq = nlInds.shape[0]
    device = nlInds.device
    bufs = th.zeros(nq,3,t,ws,ws,dtype=th.int32,device=device)

    # -- pre-computed search offsets --
    tranges,n_tranges,min_tranges = create_frame_range(t,wt,wt,pt,device)

    # -- numbify all params --
    vid_nba = cuda.as_cuda_array(vid)
    queryInds_nba = cuda.as_cuda_array(queryInds)
    nlDists_nba = cuda.as_cuda_array(nlDists)
    nlInds_nba = cuda.as_cuda_array(nlInds)
    fflow_nba = cuda.as_cuda_array(fflow)
    bflow_nba = cuda.as_cuda_array(bflow)
    bufs_nba = cuda.as_cuda_array(bufs)
    tranges_nba = cuda.as_cuda_array(tranges)
    n_tranges_nba = cuda.as_cuda_array(n_tranges)
    min_tranges_nba = cuda.as_cuda_array(min_tranges)
    # cs_nba = cuda.external_stream(cs)

    # -- launch params --
    nq = queryInds.shape[0]
    batches_per_block = 10
    bpb = batches_per_block

    # -- launch params --
    w_threads = min(ws,32)
    nthreads = (w_threads,w_threads)
    ws_iters = (ws-1)//w_threads + 1
    nblocks = (nq-1)//batches_per_block+1

    # -- exec kernel --
    numba_search[nblocks,nthreads](vid_nba,queryInds_nba,nlDists_nba,nlInds_nba,
                                   fflow_nba,bflow_nba,ps,pt,chnls,dilation,stride,
                                   bufs_nba,tranges_nba,n_tranges_nba,
                                   min_tranges_nba,ws_iters,bpb)


# @cuda.jit(debug=True,max_registers=64,opt=False)
@cuda.jit(debug=False,max_registers=64)
def numba_search(vid,queryInds,dists,inds,fflow,bflow,ps,pt,chnls,
                 dilation,stride,bufs,tranges,n_tranges,min_tranges,ws_iters,bpb):

    # -- reflective boundary [weird to match lidia] --
    def bounds(val,lim):
        # return int(val)
        if val < 0: val = (-val-1)
        elif val >= lim: val = (2*lim - val-2)
        return int(val)

    # -- reflective boundary --
    def bounds2(val,lim):
        # return int(val)
        if val < 0: val = (-val-1)
        elif val >= lim: val = (2*lim - val)
        return int(val)

    # -- reflective boundary --
    def bounds3(val,lim):
        if val < 0: val = -val-1
        elif val >= lim: val = 2*lim-val-1
        return int(val)

    # -- shapes --
    nframes,color,h,w = vid.shape
    bsize,st,ws,ws = dists.shape
    bsize,st,ws,ws,_ = inds.shape
    height,width = h,w
    Z = ps*ps*pt*chnls
    psHalf = int(ps//2)
    wsHalf = (ws-1)//2

    # -- cuda threads --
    cu_tidX = cuda.threadIdx.x
    cu_tidY = cuda.threadIdx.y
    blkDimX = cuda.blockDim.x
    blkDimY = cuda.blockDim.y
    tidX = cuda.threadIdx.x
    tidY = cuda.threadIdx.y

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
        if bidx >= queryInds.shape[0]: continue

        # -- unpack pixel locs --
        ti = queryInds[bidx,0]
        hi = queryInds[bidx,1]
        wi = queryInds[bidx,2]
        top,left = hi-psHalf,wi-psHalf

        # ---------------------------
        #     valid (anchor pixel)
        # ---------------------------

        valid_t = (ti < nframes) and (ti >= 0)
        valid_top = (hi < height) and hi >= 0
        valid_left = (wi < width) and wi >= 0
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
            # tidX = cu_tidX + _xi*blkDimX
            if tidX >= ws: continue

            for _yi in range(ws_iters):
                tidY = cu_tidY + blkDimY*_yi
                # tidY = blkDimX*cu_tidY + _yi
                if tidY >= ws: continue

                for tidZ in range(n_trange):

                    # -------------------
                    #    search frame
                    # -------------------
                    n_ti = trange[tidZ]
                    dt = trange[tidZ] - min_trange

                    # ------------------------
                    #      init direction
                    # ------------------------

                    direction = max(-1,min(1,n_ti - ti))
                    if direction != 0:

                        # -- get offset at index --
                        dtd = int(dt-direction)
                        cw0 = bufs[bidx,0,dtd,tidX,tidY]
                        ch0 = bufs[bidx,1,dtd,tidX,tidY]
                        ct0 = bufs[bidx,2,dtd,tidX,tidY]

                        # -- legalize access --
                        l_cw0 = int(max(0,min(w-1,cw0)))
                        l_ch0 = int(max(0,min(h-1,ch0)))
                        l_ct0 = int(max(0,min(ct0,nframes-1)))

                        # -- pick flow --
                        flow = fflow if direction > 0 else bflow

                        # -- access flows --
                        cw_f = cw0 + flow[l_ct0,0,l_ch0,l_cw0]
                        ch_f = ch0 + flow[l_ct0,1,l_ch0,l_cw0]

                        # -- rounding --
                        cw = max(0,min(width-1,round(cw_f)))
                        ch = max(0,min(height-1,round(ch_f)))
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

                    # --------------------
                    #      init dists
                    # --------------------
                    dist = 0

                    # -----------------
                    #    spatial dir
                    # -----------------
                    ws_i,ws_j = tidX,tidY
                    n_hi = ch + stride * (ws_i - wsHalf)
                    n_wi = cw + stride * (ws_j - wsHalf)

                    # ---------------------------
                    #      valid (search "n")
                    # ---------------------------

                    valid_t = (n_ti < nframes) and (n_ti >= 0)
                    valid_top = (n_hi < height) and (n_hi >= 0)
                    valid_left = (n_wi < width) and (n_wi >= 0)

                    valid = valid_t and valid_top and valid_left
                    valid = valid and valid_anchor
                    if not(valid): dist = np.inf

                    # ---------------------------------
                    #
                    #  compute delta over patch vol.
                    #
                    # ---------------------------------

                    for pk in range(pt):
                        for pi in range(ps):
                            for pj in range(ps):

                                # -- inside entire image --
                                vH = bounds3(hi + dilation*(pi - psHalf),height)
                                vW = bounds3(wi + dilation*(pj - psHalf),width)
                                vT = ti + pk

                                nH = bounds3(n_hi + dilation*(pi - psHalf),height)
                                nW = bounds3(n_wi + dilation*(pj - psHalf),width)
                                nT = n_ti + pk

                                # -- valid checks [for testing w/ zero pads] --
                                vvalid = (vH < height and vH >= 0)
                                vvalid = vvalid and (vW < width and vW >= 0)
                                vvalid = vvalid and (vT < nframes and vT >= 0)

                                nvalid = (nH < height and nH >= 0)
                                nvalid = nvalid and (nW < width and nW >= 0)
                                nvalid = nvalid and (nT < nframes and nT >= 0)

                                # -- all channels --
                                for ci in range(chnls):

                                    # -- get data --
                                    if vvalid:
                                        v_pix = vid[vT][ci][vH][vW]
                                    else:
                                        v_pix = 0.
                                    if nvalid:
                                        n_pix = vid[nT][ci][nH][nW]
                                    else:
                                        n_pix = 0.
                                    # v_pix = vid[vT][ci][vH][vW]
                                    # n_pix = vid[nT][ci][nH][nW]

                                    # -- compute dist --
                                    if dist < np.infty:
                                        dist += (v_pix - n_pix)**2

                    # -- dists --
                    # dist /= Z
                    # dist = dist if dist > 0 else 0
                    dists[bidx,tidZ,tidX,tidY] = dist

                    # -- inds --
                    inds[bidx,tidZ,tidX,tidY,0] = n_ti
                    inds[bidx,tidZ,tidX,tidY,1] = n_hi
                    inds[bidx,tidZ,tidX,tidY,2] = n_wi

                    # -- final check [put self@index 0] --
                    eq_ti = n_ti == ti
                    eq_hi = n_hi == hi # hi
                    eq_wi = n_wi == wi # wi
                    eq_dim = eq_ti and eq_hi and eq_wi
                    dist = dists[bidx,tidZ,tidX,tidY]
                    dists[bidx,tidZ,tidX,tidY] = -100 if eq_dim else dist



