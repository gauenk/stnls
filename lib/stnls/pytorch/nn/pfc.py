
# -- python --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- softmax --
import torch.nn.functional as nnf

# -- cpp cuda kernel --
import stnls_cuda

class PatchFCFunction(th.autograd.Function):

    # torch::Tensor vid, torch::Tensor vid_in,
    # torch::Tensor weights, torch::Tensor bias,
    # int qstart, int nqueries, int ps,
    # int top, int left, int btm, int right,
    # int start, int stride, int dilation, int adj,
    # bool only_full, bool use_reflect){

    # return PatchFCFunction.apply(vid,self.weights,self.bias,
    #                              qstart,nqueries,self.ps,
    #                              0,0,H,W,self.stride,self.dil,
    #                              adj,only_full,use_reflect)

    @staticmethod
    def forward(ctx, vid, weights, bias, bias_vid,
                qstart, nqueries, c_in, c_out, ps,
                top, left, btm, right,
                stride, dilation, adj, only_full, use_reflect):

        # -- init --
        B,T,_,H,W = vid.shape
        dtype = vid.dtype
        device = vid.device
        vid_out = th.zeros((B,T,c_out,1,H,W),device=device,dtype=dtype)
        hw_start = 0
        nqueries = T*((H-1)//stride+1)*((W-1)//stride+1)
        # print("vid.shape: ",vid.shape,nqueries)


        # -- view --
        weights = weights.view((c_out,ps,ps,c_in,ps,ps))
        bias = bias.view((c_out,ps,ps))
        # print(weights.shape)
        # print(weights)
        # torch::Tensor vid, torch::Tensor vid_in,
        # torch::Tensor weights, torch::Tensor bias,
        # int qstart, int nqueries, int ps,
        # int top, int left, int btm, int right,
        # int hw_start, int stride, int dilation, int adj,
        # bool only_full, bool use_reflect){
        # print(weights[0,:3,:3,0,0,0])
        # print(weights[0,1,0,0,:3,:3])
        # print(weights[0,0,1,0,:3,:3])

        # -- viz --
        # print("vid_out.shape: ",vid_out.shape)
        # print("vid.shape: ",vid.shape)
        # print("weights.shape: ",weights.shape)
        # print("bias.shape: ",bias.shape)
        # print("qstart,nqueries,ps,top,left,btm,right: ",
        #       qstart,nqueries,ps,top,left,btm,right)
        # print("hw_start,stride,dilation,adj,only_full,use_reflect: ",
        #       hw_start,stride,dilation,adj,only_full,use_reflect)

        # -- forward --
        # th.cuda.set_device(device)
        # print("hi.")
        stnls_cuda.pfc_forward(vid_out, vid, weights, bias,
                              qstart, nqueries, ps,
                              top, left, btm, right,
                              hw_start, stride, dilation,
                              adj, only_full, use_reflect)

        # -- for backward --
        vid_out = vid_out[:,:,:,0]
        # vid_out = vid_out.sum(3)# + bias_vid
        return vid_out

    @staticmethod
    def backward(ctx, grad_dists,inds_no_grad):
        return None

class PatchFC(th.nn.Module):

    def __init__(self, c_in, c_out, ps, stride, dil=1, device="cuda:0"):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.ps = ps
        self.dil = dil
        self.stride = stride
        self.weights,self.bias = self.init_params(c_in,c_out,ps,device)

    def set_from_fc(self,fc_layer):
        self.weights = fc_layer.weight.data
        self.bias = fc_layer.bias.data

    def init_params(self,c_in,c_out,ps,device):
        dim0 = c_in*ps*ps
        dim1 = c_out*ps*ps
        weights = th.randn((dim1,dim0),dtype=th.float32,device=device)
        bias = th.randn((dim1,),dtype=th.float32,device=device)
        weights = weights.clamp(-1,1)
        bias = bias.clamp(-1,1)
        return weights,bias

    def get_bias_vid(self,bias,H,W):
        full_pads = False
        dim = bias.shape[0]
        ps = self.ps
        stride = self.stride
        _H,_W = H,W
        sub = 1 if full_pads else (ps)
        nq = ((H-sub)//stride+1)*((W-sub)//stride+1)
        ipad = (ps)//2 if full_pads else 0
        H,W = H+2*ipad,W+2*ipad
        bias_r = repeat(bias.view(1,dim,1),'1 d 1 -> 1 d nq',nq=nq)
        ones_r = th.ones_like(bias_r)
        bias_vid = nnf.fold(bias_r,(H,W),(ps,ps),stride=stride)
        one_vid = nnf.fold(ones_r,(H,W),(ps,ps),stride=stride)
        if ipad > 0: bias_vid /= one_vid
        bias_vid = bias_vid[...,:_H,:_W] / one_vid[...,:_H,:_W]
        return bias_vid

    def forward(self, vid, qstart=0, nqueries=-1):
        adj = -self.ps//2
        only_full = False
        use_reflect = True
        H,W = vid.shape[-2:]
        bias_vid = None#self.get_bias_vid(self.bias,H,W)
        # exit(0)
        assert vid.shape[-3] == self.c_in
        return PatchFCFunction.apply(vid,self.weights,self.bias,
                                     bias_vid,qstart,nqueries,
                                     self.c_in,self.c_out,self.ps,
                                     0,0,H,W,self.stride,self.dil,
                                     adj,only_full,use_reflect)

    def flops(self,nsearch,T,C,H,W,inds_k):

        # -- init --
        flops = 0

        return flops
