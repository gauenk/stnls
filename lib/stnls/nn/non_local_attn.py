"""

The Non-Local Attention Module


"""

# -- torch network deps --
import stnls
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- config --
import copy
dcopy = copy.deepcopy
from stnls.utils import config
from stnls.utils.misc import optional

# -- from timm.models.layers import trunc_normal_ --
import torch
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out

# -- rescale flow --
import torch.nn.functional as tnnf

# -- benchmarking --
from stnls.utils.timer import ExpTimer,ExpTimerList

# -- layers --
from .utils import rescale_flows

def default_pairs():
    pairs = {"nheads":1,"inner_mult":1,
             "embed_dim":16,
             "qk_frac":1.,"qkv_bias":True,
             "qkv_ngroups":1,
             "use_attn_projection":True,
             "drop_rate_proj":0.,
             "attn_timer":False,"use_attn_flow":True,
             "use_norm_layer":False}
    return pairs

def extract_config(cfg,restrict=True):
    cfg = config.extract_pairs(cfg,default_pairs(),restrict=restrict)
    return cfg

class NonLocalAttention(nn.Module):

    def __init__(self, attn_cfg, search_cfg, normz_cfg, agg_cfg):
        super().__init__()

        # -- attn_cfg defaults --
        # attn_cfg = dcopy(attn_cfg)
        # pairs = {"use_norm_layer":False}
        # for k,v in pairs.items():
        #     if not(k in attn_cfg):
        #         attn_cfg[k] = v

        # -- unpack --
        nheads = attn_cfg.nheads
        inner_mult = optional(attn_cfg,"inner_mult",1)
        embed_dim = attn_cfg.embed_dim * inner_mult
        io_dim = attn_cfg.embed_dim * nheads

        # -- init configs --
        self.dim = io_dim
        self.attn_cfg = attn_cfg
        self.search_cfg = search_cfg
        self.normz_cfg = normz_cfg
        self.agg_cfg = agg_cfg

        # -- init attn fxns --
        self.search = stnls.search.init(search_cfg)
        self.normz = stnls.normz.init(normz_cfg)
        self.agg = stnls.agg.init(agg_cfg)

        # -- init vars of interest --
        self.use_norm_layer = attn_cfg.use_norm_layer
        self.use_flow = attn_cfg.use_attn_flow
        self.use_state_update = search_cfg.use_state_update
        self.ps = search_cfg.ps
        self.search_name = search_cfg.search_name
        self.stride0 = search_cfg.stride0
        self.dilation = search_cfg.dilation
        self.k_agg = search_cfg.k_agg

        # -- qkv attn --
        self.qkv = ConvQKV(io_dim,nheads,embed_dim,
                           attn_cfg.qk_frac,bias=attn_cfg.qkv_bias,
                           ngroups=attn_cfg.qkv_ngroups)

        # -- projection layer --
        if attn_cfg.use_attn_projection:
            self.proj = nn.Conv2d(io_dim*inner_mult,io_dim,(1,1),stride=1,
                                  padding="same",groups=1)
            self.proj_drop = nn.Dropout(attn_cfg.drop_rate_proj)
        else:
            self.proj = nn.Identity()
            self.proj_drop = nn.Identity()

        # -- normzliation --
        self.norm_layer = LayerNorm2D(io_dim) if self.use_norm_layer else nn.Identity()

        # -- timers --
        self.use_timer = attn_cfg.attn_timer
        self.times = ExpTimerList(attn_cfg.attn_timer)
        self.timer = None

    def forward(self, vid, flows=None, state=None):

        # -- init timer --
        self.timer = ExpTimer(self.use_timer)
        self.timer.sync_start("attn")

        # -- update flow --
        B,T,C,H,W = vid.shape
        if self.use_flow: flows = rescale_flows(flows,H,W)

        # -- extract --
        in_vid = vid
        vid = self.norm_layer(vid)
        q_vid,k_vid,v_vid = self.get_qkv(vid)

        # -- search --
        dists,inds = self.run_search(q_vid,k_vid,flows,state)

        # -- normalize --
        weights,inds = self.run_normalize(dists,inds)

        # -- aggregate --
        vid = self.run_aggregation(v_vid,weights,inds,vid.shape)

        # -- projection --
        vid = self.run_projection(vid)

        # -- timing --
        self.timer.sync_stop("attn")
        if self.use_timer:
            self.times.update_times(self.timer)

        return vid

    def get_qkv(self,vid):

        # -- compute --
        B, T, C, H, W = vid.shape
        vid = vid.view(B*T,C,H,W)
        q_vid, k_vid, v_vid = self.qkv(vid,None)

        # -- reshape --
        q_vid = q_vid.view(B,T,-1,H,W)
        k_vid = k_vid.view(B,T,-1,H,W)
        v_vid = v_vid.view(B,T,-1,H,W)

        return q_vid,k_vid,v_vid

    def run_search(self,q_vid,k_vid,flows,state):
        self.timer.sync_start("search")
        if self.search_name == "refine":
            inds_p = self.inds_rs1(state[0])
            dists,inds = self.search(q_vid,k_vid,inds_p)
        elif self.search_name == "rand_inds":
            dists,inds = self.search(q_vid,k_vid)
        else:
            dists,inds = self.search(q_vid,k_vid,flows.fflow,flows.bflow)
        self.update_state(state,dists,inds,q_vid.shape)
        self.timer.sync_stop("search")
        return dists,inds

    def run_normalize(self,dists,inds):
        self.timer.sync_start("normz")
        dists,inds = self.normz(dists,inds)
        self.timer.sync_stop("normz")
        return dists,inds

    def run_aggregation(self,v_vid,dists,inds,vshape):

        # -- aggregate patches --
        self.timer.sync_start("agg")
        vid = self.agg(v_vid,dists,inds)
        self.timer.sync_stop("agg")

        return vid

    def run_projection(self,vid):
        self.timer.sync_start("trans")
        B = vid.shape[0]
        vid = rearrange(vid,'b t c h w -> (b t) c h w')
        vid = self.proj(vid)
        vid = self.proj_drop(vid)
        vid = rearrange(vid,'(b t) c h w -> b t c h w',b=B)
        self.timer.sync_stop("trans")
        return vid

    def update_state(self,state,dists,inds,vshape):
        if not(self.use_state_update): return
        T,C,H,W = vshape[-4:]
        nH = (H-1)//self.stride0+1
        nW = (W-1)//self.stride0+1
        state[1] = state[0]
        state[0] = self.inds_rs0(inds.detach(),nH,nW)

    def inds_rs0(self,inds,nH,nW):
        if not(inds.ndim == 5): return inds
        rshape = 'b h (T nH nW) k tr -> T nH nW b h k tr'
        inds = rearrange(inds,rshape,nH=nH,nW=nW)
        return inds

    def inds_rs1(self,inds):
        if not(inds.ndim == 7): return inds
        rshape = 'T nH nW b h k tr -> b h (T nH nW) k tr'
        inds = rearrange(inds,rshape)
        return inds

    def extra_repr(self) -> str:
        str_repr = "Attention: \n" + str(self.attn_cfg) + "\n"*2
        str_repr += "Search: \n" + str(self.search_cfg) + "\n"*2
        str_repr += "Reduce: \n" + str(self.agg_cfg) + "\n"*2
        return str_repr

    def flops(self, H, W):

        # -- init flops --
        flops = 0

        # -- num of reference points --
        nrefs = ((H-1)//self.stride0+1) * ((W-1)//self.stride0+1)

        # -- convolution flops --
        flops += self.qkv.flops(H,W)
        # print("product: ",self.qkv.flops(H,W))

        # -- non-local search --
        C = self.qkv.to_q.out_channels
        vshape = (1,C,H,W)
        flops += self.search.flops(1,C,H,W)

        # -- normalize --
        flops += self.normz.flops()

        # -- weighted patch sum --
        flops += self.agg.flops()

        # -- projection --
        flops += nrefs * self.dim * self.dim

        return flops

    def reset_times(self):
        self.times = ExpTimerList(self.use_timer)


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Feature Transforms
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class ConvQKV(nn.Module):
    def __init__(self, input_dim, heads = 8, dim_head = 64, qk_frac=1.,
                 kernel_size=1,q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 ngroups=1, last_stage=False,bias=True):

        super().__init__()

        inner_dim = dim_head *  heads
        inner_dim_qk = max(int(qk_frac*dim_head),1) * heads
        self.heads = heads
        pad = (kernel_size - q_stride)//2
        self.to_q = nn.Conv2d(input_dim, inner_dim_qk, kernel_size=kernel_size,
                              stride=q_stride, padding=pad, bias=bias,
                              groups=ngroups,padding_mode="reflect")
        self.to_k = nn.Conv2d(input_dim, inner_dim_qk, kernel_size=kernel_size,
                              stride=k_stride, padding=pad, bias=bias,
                              groups=ngroups,padding_mode="reflect")
        self.to_v = nn.Conv2d(input_dim, inner_dim, kernel_size=kernel_size,
                              stride=v_stride, padding=pad, bias=bias,
                              groups=ngroups,padding_mode="reflect")

    def forward(self, x, attn_kv=None):

        # -- unpack --
        b, c, h, w = x.shape
        nheads = self.heads
        attn_kv = x if attn_kv is None else attn_kv

        # -- forward --
        q = self.to_q(x)
        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)

        return q,k,v

    def flops(self, H, W):
        flops = 0
        flops += conv2d_flops(self.to_q,H,W)
        flops += conv2d_flops(self.to_k,H,W)
        flops += conv2d_flops(self.to_v,H,W)
        return flops

def conv2d_flops(conv,H,W):

    # -- unpack --
    ksize = conv.kernel_size
    stride = conv.stride
    groups = conv.groups
    # W = conv.weights
    # b = conv.bias
    in_C = conv.in_channels
    out_C = conv.out_channels

    # -- flop --
    flop = (H // stride[0]) * (W // stride[1]) * (ksize[0] * ksize[1])
    flop *= ((in_C//groups) * (out_C//groups) * groups)
    return flop


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


class LayerNorm2D(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    """ copied from https://github.com/rwightman/pytorch-image-models/blob/d7b55a9429f3d56a991e604cbc2e9fdf1901612f/timm/models/layers/norm.py#L26 """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self,vid: torch.Tensor) -> torch.Tensor:
        B,T = vid.shape[:2]
        vid = rearrange(vid,'b t c h w -> (b t) c h w ')
        vid = F.layer_norm(vid.permute(0, 2, 3, 1), self.normalized_shape,
                           self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        vid = rearrange(vid,'(b t) c h w -> b t c h w ',b=B)
        vid = vid.contiguous()
        return vid
