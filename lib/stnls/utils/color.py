
import torch as th
import numpy as np
from einops import rearrange

def rgb2gray():
    nn_rgb2gray = th.nn.Conv2d(in_channels=3, out_channels=1,
                               kernel_size=(1, 1), bias=False)
    nn_rgb2gray.weight.data = th.tensor([0.2989, 0.5870, 0.1140],
                                        dtype=th.float32).view(1, 3, 1, 1)
    nn_rgb2gray.weight.requires_grad = False
    return nn_rgb2gray

def exec_rgb2gray(tensor_rgb,inplace=False):
    nn_rgb2gray = rgb2gray().to(tensor_rgb.device)
    nn_rgb2gray.weight.data = nn_rgb2gray.weight.data.type(tensor_rgb.dtype)
    if inplace:
        raise NotImplemented("")
    else:
        tensor_gray = nn_rgb2gray(tensor_rgb)
    return tensor_gray

def yuv2rgb_patches(patches):
    patches_rs = rearrange(patches,'b k pt c ph pw -> (b k pt) c ph pw')
    yuv2rgb(patches_rs)

def yuv2rgb(burst):
    # -- weights --
    t,c,h,w = burst.shape
    w = [1./np.sqrt(3),1./np.sqrt(2),np.sqrt(2.)/np.sqrt(3)]
    # -- copy channels --
    y,u,v = t_copy(burst[:,0]),t_copy(burst[:,1]),t_copy(burst[:,2])
    # -- yuv -> rgb --
    burst[:,0,...] = w[0] * y + w[1] * u + w[2] * 0.5 * v
    burst[:,1,...] = w[0] * y - w[2] * v
    burst[:,2,...] = w[0] * y - w[1] * u + w[2] * 0.5 * v

def rgb2yuv(burst):
    # -- weights --
    t,c,h,w = burst.shape
    # -- copy channels --
    r,g,b = t_copy(burst[:,0]),t_copy(burst[:,1]),t_copy(burst[:,2])
    # -- yuv -> rgb --
    weights = [1./np.sqrt(3),1./np.sqrt(2),np.sqrt(2.)*2./np.sqrt(3)]
    # -- rgb -> yuv --
    burst[:,0,...] = weights[0] * (r + g + b)
    burst[:,1,...] = weights[1] * (r - b)
    burst[:,2,...] = weights[2] * (.25 * r - 0.5 * g + .25 * b)


def t_copy(tensor):
    if th.is_tensor(tensor):
        return tensor.clone()
    else:
        return tensor.copy()
