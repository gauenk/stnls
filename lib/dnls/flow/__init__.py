"""
Wrap the opencv optical flow

"""
# -- linalg --
import numpy as np
import torch as th
# import jax.numpy as jnp
from einops import rearrange,repeat

# -- misc --
from easydict import EasyDict as edict

# -- opencv --
with_cv = False
try:
    import cv2
    with_cv = True
except:
    pass

# -- local --
from ..utils import color

def get_flow_batch(run_flow,use_clean,noisy,clean,sigma):
    if run_flow is True and with_cv is True:
        if use_clean: vid = noisy
        else: vid = clean
        B = vid.shape[0]
        flows = edict()
        flows.fflow,flows.bflow = [],[]
        for b in range(B):
            flows_b = run(vid[b],0.)
            flows.fflow.append(flows_b.fflow)
            flows.bflow.append(flows_b.bflow)
        flows.fflow = th.stack(flows.fflow)
        flows.bflow = th.stack(flows.bflow)
        return flows
    else:
        if th.is_tensor(noisy): device = noisy.device
        else: device = None
        flows = init_flows_batch(noisy.shape,device)
        return flows


def get_flow(run_flow,use_clean,noisy,clean,sigma):
    if run_flow is True and with_cv is True:
        if use_clean:
            return run(clean,0.)
        else:
            return run(noisy,sigma)
    else:
        if th.is_tensor(noisy): device = noisy.device
        else: device = None
        flows = init_flows(noisy.shape,device)
        return flows

def init_flows_batch(vshape,device):
    b,t,c,h,w = vshape
    flows = edict()
    flows.fflow = np.zeros((b,t,2,h,w),dtype=np.float32)
    flows.bflow = np.zeros((b,t,2,h,w),dtype=np.float32)
    if not(device is None):
        flows.fflow = th.from_numpy(flows.fflow).to(device)
        flows.bflow = th.from_numpy(flows.bflow).to(device)
    return flows


def init_flows(vshape,device):
    t,c,h,w = vshape
    flows = edict()
    flows.fflow = np.zeros((t,2,h,w),dtype=np.float32)
    flows.bflow = np.zeros((t,2,h,w),dtype=np.float32)
    if not(device is None):
        flows.fflow = th.from_numpy(flows.fflow).to(device)
        flows.bflow = th.from_numpy(flows.bflow).to(device)
    return flows

def run(vid_in,sigma,use_copy=False):

    # -- to numpy --
    device = vid_in.device
    if th.is_tensor(vid_in):
        vid_in = vid_in.cpu().numpy()

    #-- copy --
    # copy data for no-rounding-error from RGB <-> YUV
    if use_copy:
        vid = vid_in.copy()
    else:
        vid = vid_in

    # -- alloc --
    flows = init_flows(vid.shape,None)

    # -- color2gray --
    t,c,h,w = vid.shape
    vid = np.clip(vid,0,255.).astype(np.uint8)
    color.rgb2yuv(vid)
    vid = vid[:,[0],:,:] # only Y
    vid = rearrange(vid,'t c h w -> t h w c')

    # -- computing --
    for ti in range(t-1):
        flows.fflow[ti] = pair2flow(vid[ti],vid[ti+1])
    for ti in reversed(range(t-1)):
        flows.bflow[ti] = pair2flow(vid[ti+1],vid[ti])

    # -- packing --
    flows.fflow = th.from_numpy(flows.fflow).to(device)
    flows.bflow = th.from_numpy(flows.bflow).to(device)

    # -- gray2color --
    color.yuv2rgb(vid)

    return flows


def pair2flow(frame_a,frame_b,bound=15):

    # -- create flow object --
    dtvl1_flow = cv2.optflow.DualTVL1OpticalFlow_create()

    # -- exec --
    flow = dtvl1_flow.calc(frame_a, frame_b, None)

    # -- format --
    flow = flow.transpose(2,0,1)

    return flow


def pair2flow_gpu(frame_a,frame_b,device):

    # -- create opencv-gpu frames --
    gpu_frame_a = cv.cuda_GpuMat()
    gpu_frame_b = cv.cuda_GpuMat()
    gpu_frame_a.upload(frame_a.cpu().numpy())
    gpu_frame_b.upload(frame_b.cpu().numpy())

    # -- create flow object --
    gpu_flow = cv.cuda_FarnebackOpticalFlow.create(5, 0.5, False,
                                                   15, 3, 5, 1.2, 0)
    # -- exec flow --
    flow = cv.cuda_FarnebackOpticalFlow.calc(gpu_flow, gpu_frame_a,
                                             gpu_frame_b, None)
    flow = flow.download()
    flow = flow.transpose(2,0,1)
    flow = th.from_numpy(flow).to(device)

    return flow

def flows2vids(flows):
    vid = dict.fromkeys(flows)
    for key in flows:
        _flows = flows[key]
        device = None
        if th.is_tensor(_flows):
            device = _flows.device
            _flows = _flows.cpu().numpy()
        vid[key] = flow2vid(_flows)
        if not(device is None):
            vid[key] = th.from_numpy(vid[key]).to(device)
    return edict(vid)

def flow2vid(flow):
    t,_,h,w = flow.shape
    mask = np.zeros((t,3,h,w),dtype=np.uint8)
    mask[:,1] = 255. # saturation to max
    t = len(flow)
    for ti in range(t):

        # Computes the magnitude and angle of the 2D vectors
        print(flow[ti].shape)
        magnitude, angle = cv2.cartToPolar(flow[ti,0], flow[ti,1])

        # Sets image hue according to the optical flow
        # direction
        mask[ti,0] = angle * 180 / np.pi / 2

        # Sets image value according to the optical flow
        # magnitude (normalized)
        mask[ti,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Converts HSV to RGB (BGR) color representation
        mask_ti = rearrange(mask[ti],'c h w -> h w c')
        mask_ti = cv2.cvtColor(mask_ti, cv2.COLOR_HSV2BGR)
        mask_ti = rearrange(mask_ti,'h w c -> c h w')
        mask[ti] = mask_ti
    return mask

def pth2jax(flows):
    flows_jax = {}
    for key in flows:
        flows_jax[key] = jnp.array(flows[key].cpu().numpy())
        # edict({jnp.array(flows[key].cpu().numpy()) for key in flows})
    flows_jax = edict(flows_jax)
    return flows_jax
