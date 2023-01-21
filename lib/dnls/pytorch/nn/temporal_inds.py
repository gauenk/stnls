"""

Get temporal inds from spatial inds

"""

def run(inds,flows,wt):


    # -- shaping --
    B,Q,K,_ = inds.shape
    T = flows.fflow.shape[0]
    st = 2*wt + 1
    nT = min(st,T)

    # -- allocate --
    inds_t = -th.ones((B,Q,K,nT,3),device=inds.device,dtype=inds.dtype)

    # -- run --
    fflow = flows.fflow
    bflow = flows.bflow
    dnls_cuda.temporal_inds(inds,fflow,bflow,pfflow,inds_t)

    # -- format --

    return inds_t
