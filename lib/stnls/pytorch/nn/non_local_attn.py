"""

The Non-Local Attention Module


"""

# -- torch network deps --
import stnls
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import unfold
from einops import rearrange,repeat

# -- config --
import copy
dcopy = copy.deepcopy

# -- from timm.models.layers import trunc_normal_ --
import torch
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out

# -- rescale flow --
from dev_basics import flow

# -- benchmarking --
from dev_basics.utils.timer import ExpTimer,ExpTimerList

class NonLocalAttention(nn.Module):

    def __init__(self, attn_cfg, search_cfg, normz_cfg, agg_cfg):
        super().__init__()

        # -- attn_cfg defaults --
        attn_cfg = dcopy(attn_cfg)
        pairs = {"use_norm_layer":False}
        for k,v in pairs.items():
            if not(k in attn_cfg):
                attn_cfg[k] = v

        # -- unpack --
        embed_dim = attn_cfg.embed_dim
        nheads = attn_cfg.nheads
        dim = embed_dim * nheads

        # -- init configs --
        self.dim = dim
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
        self.use_flow = attn_cfg.use_flow
        self.use_state_update = attn_cfg.use_state_update
        self.ps = search_cfg.ps
        self.search_name = search_cfg.search_name
        self.stride0 = search_cfg.stride0
        self.dilation = search_cfg.dilation
        self.k_s = search_cfg.k
        self.k_n = normz_cfg.k_n
        self.k_a = agg_cfg.k_a

        # -- attn init --
        self.token_projection = attn_cfg.token_projection
        self.qkv = ConvQKV(dim,nheads,embed_dim,
                           attn_cfg.qk_frac,bias=attn_cfg.qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_cfg.drop_rate_proj)
        self.norm_layer = LayerNorm2D(dim) if self.use_norm_layer else nn.Identity()

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
        if self.use_flow: flows = flow.rescale_flows(flows,H,W)

        # -- extract --
        vid = self.norm_layer(vid)
        q_vid,k_vid,v_vid = self.get_qkv(vid)

        # -- search --
        dists,inds = self.run_search(q_vid,k_vid,flows,state)
        # th.cuda.synchronize()

        # -- normalize --
        dists = self.run_normalize(dists)

        # -- aggregate --
        patches = self.run_aggregation(v_vid,dists,inds)

        # -- transform --
        patches = self.run_transform(patches)

        # -- fold --
        vid = self.run_fold(patches,vid.shape)

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

    def run_normalize(self,dists):
        self.timer.sync_start("normz")
        dists = self.normz(dists)
        self.timer.sync_stop("normz")
        return dists

    def run_aggregation(self,v_vid,dists,inds):
        self.timer.sync_start("agg")
        patches = self.agg(v_vid,dists,inds)
        self.timer.sync_stop("agg")
        return patches

    def run_transform(self,patches):
        self.timer.sync_start("trans")
        patches = self.proj(patches)
        patches = self.proj_drop(patches)
        self.timer.sync_stop("trans")
        return patches

    def run_fold(self,patches,vshape):

        # -- timing --
        self.timer.sync_start("fold")

        # -- init folding --
        B,ps = vshape[0],self.search_cfg.ps
        fold = stnls.iFoldz(vshape,None,stride=self.stride0,
                            dilation=self.dilation,adj=0,only_full=False,
                            use_reflect=True,device=patches.device)

        # -- reshape for folding --
        shape_str = '(b q ph pw) c -> b q 1 1 c ph pw'
        patches = rearrange(patches,shape_str,b=B,ph=ps,pw=ps)
        patches = patches.contiguous()

        # -- fold --
        fold(patches)

        # -- unpack --
        vid = fold.vid / fold.zvid

        # -- debug --
        any_nan = th.any(th.isnan(vid)).item()
        if any_nan:
            any_fold_nan = th.any(th.isnan(fold.vid)).item()
            any_patch_nan = th.any(th.isnan(fold.vid)).item()
            any_zero = th.any(th.abs(fold.zvid)<1e-10).item()
            print("[%s] found a nan!: " % __file__,any_nan,any_zero,
                  any_fold_nan,any_patch_nan)
            print(self.search_name)
            exit(0)

        # -- timing --
        self.timer.sync_stop("fold")

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

    def get_patches(self,vid):
        vid = rearrange(vid,'B T C H W -> (B T) C H W')
        patches = unfold(vid,(self.ps,self.ps))
        patches = rearrange(patches,'b (p2 d) n -> (b n p2) 1 d',d=self.dim)
        return patches

    def extra_repr(self) -> str:
        str_repr = "Attention: \n" + str(self.attn_cfg) + "\n"*2
        str_repr += "Search: \n" + str(self.search_cfg) + "\n"*2
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

        # -- fold --
        ps = self.search_cfg.ps
        flops += nrefs * ps * ps
        # print(flops)

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
                 last_stage=False,bias=True):

        super().__init__()

        inner_dim = dim_head *  heads
        inner_dim_qk = int(qk_frac*dim_head) * heads
        self.heads = heads
        pad = (kernel_size - q_stride)//2
        self.to_q = nn.Conv2d(input_dim, inner_dim_qk, kernel_size=kernel_size,
                              stride=q_stride, padding=pad, bias=bias,
                              groups=1,padding_mode="reflect")
        self.to_k = nn.Conv2d(input_dim, inner_dim_qk, kernel_size=kernel_size,
                              stride=k_stride, padding=pad, bias=bias,
                              groups=1,padding_mode="reflect")
        self.to_v = nn.Conv2d(input_dim, inner_dim, kernel_size=kernel_size,
                              stride=v_stride, padding=pad, bias=bias,
                              groups=1,padding_mode="reflect")

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