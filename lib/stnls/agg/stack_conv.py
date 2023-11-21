"""

  Combine NonLocalStack with Convolution

"""

# -- python --
import torch as th
import torch.nn as nn
from einops import rearrange
from stnls.utils import extract_pairs

# -- stacking --
from .non_local_stack import extract_config as extract_config_stack
from .non_local_stack import init as init_stack

# -- projection --
from .proj_menu import extract_config as extract_config_proj
from .proj_menu import init as init_proj

class StackConv(nn.Module):

    def __init__(self,stacker,proj,proj_drop,proj_version):
        super().__init__()
        self.stacker = stacker
        self.proj = proj
        self.proj_drop = proj_drop
        self.proj_version = proj_version

    def forward(self,vid,weights,flows):
        stack = self.stacker(vid,weights,flows)
        stack = rearrange(stack,'b hd k t c h w -> b t k (hd c) h w')
        vid = self.run_projection(stack)
        # print("[stack_conv] vid.shape: ",vid.shape)
        return vid

    def run_projection(self,stack):
        B = stack.shape[0]
        if self.proj_version in ["v1","v2","v3"]:
            stack = rearrange(stack,'b t k c h w -> (b t) c k h w')
            stack = self.proj(stack)
            stack = self.proj_drop(stack)
            stack = th.mean(stack,2,keepdim=True)
        else:
            stack = self.proj(stack)
            stack = self.proj_drop(stack)
        vid = rearrange(stack,'(b t) c 1 h w -> b t c h w',b=B)
        return vid


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#            [Functional API]  stnls.agg.stackconv(...)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def _apply(vid, weights, flows, ps=1, stride0=1, pt=1,
           reflect_bounds=True, dilation=1, use_adj=False, itype="float"):
    # wrap "new (2018) apply function
    # https://discuss.pytorch.org #13845/17
    # cfg = extract_config(kwargs)
    fxn = NonLocalSearchFunction.apply
    return fxn(vid,weights,flows,ps,stride0,pt,reflect_bounds,dilation,use_adj,itype)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#        [Python Dict API] stnls.agg.init(pydict)
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_config(_cfg,restrict=True):

    # -- get stack config --
    stack_cfg = extract_config_stack(_cfg,restrict=restrict)
    proj_cfg = extract_config_proj(_cfg,restrict=restrict)

    # -- append "StackConv" keys --
    stackconv_pairs = {key:val for key,val in stack_cfg.items()}
    for key in proj_cfg: stackconv_pairs[key] = proj_cfg[key]

    # -- extract with full defaults --
    cfg = extract_pairs(_cfg,stackconv_pairs,restrict=restrict)

    return cfg

def init(cfg):
    cfg = extract_config(cfg,False)
    stacker = init_stack(cfg)
    proj,proj_drop = init_proj(cfg)
    search = StackConv(stacker,proj,proj_drop,cfg.nlstack_proj_version)
    return search



