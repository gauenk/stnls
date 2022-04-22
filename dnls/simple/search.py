
# -- python-only kernel --
from numba import cuda,jit

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

def run(vid,queryInds,flow,k,ps,pt,ws,wt,chnls):

    # -- allocate --
    device = queryInds.device
    nq = queryInds.shape[0]
    ns = ws * ws * (2 * wt + 1)
    nlDists,nlInds = allocate(nq,k,device)
    nlDists_exh,nlInds_exh = allocate(nq,ns,device)

    # -- unpack --
    fflow = flow.fflow
    bflow = flow.bflow

    # -- exec --
    numba_search_launcher(vid,queryInds,nlDists_exh,nlInds_exh,
                          fflow,bflow,k,ps,pt,ws,wt,chnls)

    # -- patches of topk --
    get_topk(nlDists_exh,nlInds_exh,nlDists,nlInds)

    return nlDists,nlInds

def allocate(nq,ns,device):
    dists = th.zeros((nq,ns),device=device,dtype=th.float32)
    dists[...] = float("inf")
    inds = th.zeros((nq,ns,3),device=device,dtype=th.int32)
    inds[...] = -1
    return dists,inds

def get_topk(l2_vals,l2_inds,vals,inds):

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
                          fflow,bflow,k,ps,pt,ws,wt,chnls):

    # -- reshape for easy indexing --
    nlDists = rearrange(nlDists,'q (sx sy st) -> q sx sy st',sx=ws,sy=ws)
    nlInds = rearrange(nlInds,'q (sx sy st) a -> q sx sy st a',sx=ws,sy=ws)
    nlDists = nlDists.contiguous()
    nlInds = nlInds.contiguous()

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
                                   fflow_nba,bflow_nba,ps,pt,chnls,
                                   bufs_nba,tranges_nba,n_tranges_nba,
                                   min_tranges_nba,ws_iters,bpb)


# @cuda.jit(debug=True,max_registers=64,opt=False)
@cuda.jit(debug=False,max_registers=64)
def numba_search(vid,queryInds,dists,inds,fflow,bflow,ps,pt,chnls,
                 bufs,tranges,n_tranges,min_tranges,ws_iters,bpb):

    # -- reflective boundary --
    def bounds(val,lim):
        if val < 0: val = (-val-1)
        if val >= lim: val = (2*lim - val - 1)
        return val

    # -- shapes --
    nframes,color,h,w = vid.shape
    bsize,st,sx,sy = dists.shape
    bsize,st,sx,sy,_ = inds.shape
    height,width = h,w
    Z = ps*ps*pt*chnls
    ws = sx
    psHalf = int(ps//2)

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

        ti = queryInds[bidx,0]
        hi = queryInds[bidx,1]
        wi = queryInds[bidx,2]
        # ti,hi,wi = 0,0,0
        top,left = hi-psHalf,wi-psHalf

        # ---------------------------
        #     valid (anchor pixel)
        # ---------------------------

        valid_t = (ti+pt-1) < nframes
        valid_t = valid_t and (ti >= 0)
        valid_top = hi < height and hi >= 0
        valid_left = wi < width and wi >= 0
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
                        dtd = dt-direction
                        # if dtd >= bufs.shape[2]: continue
                        cw0 = bufs[bidx,0,dt-direction,tidX,tidY]
                        ch0 = bufs[bidx,1,dt-direction,tidX,tidY]
                        ct0 = bufs[bidx,2,dt-direction,tidX,tidY]

                        flow = fflow if direction > 0 else bflow

                        cw_f = cw0 + flow[ct0,0,ch0,cw0]
                        ch_f = ch0 + flow[ct0,1,ch0,cw0]

                        cw = max(0,min(w-1,round(cw_f)))
                        ch = max(0,min(h-1,round(ch_f)))
                        ct = n_ti
                    else:
                        cw = wi
                        ch = hi
                        ct = ti

                    # ----------------
                    #     update
                    # ----------------
                    # if dt >= bufs.shape[2]: continue
                    bufs[bidx,0,dt,tidX,tidY] = cw#cw_vals[ti-direction]
                    bufs[bidx,1,dt,tidX,tidY] = ch#ch_vals[t_idx-direction]
                    bufs[bidx,2,dt,tidX,tidY] = ct#ct_vals[t_idx-direction]

                    # --------------------
                    #      init dists
                    # --------------------
                    dist = 0

                    # --------------------------------
                    #   search patch's top,left
                    # --------------------------------

                    # -- target pixel we are searching --
                    if (n_ti) < 0: dist = np.inf
                    if (n_ti) >= (nframes-pt+1): dist = np.inf

                    # -----------------
                    #    spatial dir
                    # -----------------

                    # shift_w = min(0,cw - (ws-1)//2) \
                    #     + max(0,cw + (ws-1)//2 - w  + ps)
                    # shift_h = min(0,ch - (ws-1)//2) \
                    #     + max(0,ch + (ws-1)//2 - h  + ps)
                    shift_w = 0
                    shift_h = 0

                    # -- spatial endpoints --
                    sh_start = ch - (ws-1)//2#max(0,ch - (ws-1)//2 - shift_h)
                    # sh_end = min(h-psHalf,ch + (ws-1)//2 - shift_h)+1

                    sw_start = cw - (ws-1)//2#max(0,cw - (ws-1)//2 - shift_w)
                    # sw_end = min(w-ps,cw + (ws-1)//2 - shift_w)+1

                    n_top = sh_start + tidX
                    n_left = sw_start + tidY
                    n_hi = n_top + psHalf
                    n_wi = n_left + psHalf

                    # ---------------------------
                    #      valid (search "n")
                    # ---------------------------

                    valid_t = ((n_ti+pt-1) < nframes) and (n_ti >= 0)
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

                    # -- compute difference over patch volume --
                    for pk in range(pt):
                        for pi in range(ps):
                            for pj in range(ps):

                                # -- inside entire image --
                                vH = bounds(top+pi,h)
                                vW = bounds(left+pj,w)
                                vT = ti + pk

                                nH = bounds(n_top+pi,h)
                                nW = bounds(n_left+pj,w)
                                nT = n_ti + pk

                                # -- all channels --
                                for ci in range(chnls):

                                    # -- get data --
                                    v_pix = vid[vT][ci][vH][vW]/255.
                                    n_pix = vid[nT][ci][nH][nW]/255.

                                    # -- compute dist --
                                    if dist < np.infty:
                                        dist += (v_pix - n_pix)**2

                    # -- dists --
                    dist /= Z
                    dist = dist if dist > 0 else 0
                    dists[bidx,tidZ,tidX,tidY] = dist

                    # -- inds --
                    inds[bidx,tidZ,tidX,tidY,0] = n_ti
                    inds[bidx,tidZ,tidX,tidY,1] = n_top + psHalf
                    inds[bidx,tidZ,tidX,tidY,2] = n_left + psHalf

                    # -- final check [put self@index 0] --
                    eq_ti = n_ti == ti
                    eq_hi = n_top == top # hi
                    eq_wi = n_left == left # wi
                    eq_dim = eq_ti and eq_hi and eq_wi
                    dist = dists[bidx,tidZ,tidX,tidY]
                    dists[bidx,tidZ,tidX,tidY] = -100 if eq_dim else dist



