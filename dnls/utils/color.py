
import torch as th

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

