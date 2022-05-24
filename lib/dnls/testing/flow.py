
# -- data mgnmnt --
from easydict import EasyDict as edict

# -- linalg --
import numpy as np
import torch as th

# -- import for optical flow --
allow_of = True
try:
    import svnlb
except:
    print("No optical flow available. All flows are zero.")
    allow_of = False

def get_flow(comp_flow,use_clean,noisy,clean,sigma):

    # -- params --
    if use_clean:
        flow_img = clean
        sigma = 0.
    else: flow_img = noisy

    # -- exec --
    fflow,bflow = compute_flow(comp_flow,flow_img,sigma)

    # -- pack --
    flows = edict()
    flows.fflow = fflow
    flows.bflow = bflow
    return flows

    return flow

def compute_flow(comp_flow,burst,sigma):
    device = burst.device
    if comp_flow and allow_of:
        #  -- TV-L1 Optical Flow --
        flow_params = {"nproc":0,"tau":0.25,"lambda":0.2,"theta":0.3,
                       "nscales":100,"fscale":1,"zfactor":0.5,"nwarps":5,
                       "epsilon":0.01,"verbose":False,"testing":False,'bw':True}
        fflow,bflow = svnlb.swig.runPyFlow(burst,sigma,flow_params)
    else:
        #  -- Empty shells --
        t,c,h,w = burst.shape
        tf32,tfl = th.float32,th.long
        fflow = th.zeros(t,2,h,w,dtype=tf32,device=device)
        bflow = fflow.clone()
    return fflow,bflow

