"""

Search using randomized inds

"""

# -- pytorch --
import torch as th
import torch.nn as nn

# -- local --
from .non_local_search import init as init_nls
from .refinement import init as init_refine
from .non_local_search import extract_config as extract_config_nls
from .refinement import extract_config as extract_config_refine


class RandIndsSearch(th.nn.Module):

    def __init__(self, nls, refine):
        super().__init__()
        self.nls = nls
        self.refine = refine

    def forward(self,vid0,vid1,batchsize=-1):
        rand0 = th.randn_like(vid0)
        rand1 = th.randn_like(vid1)
        B,T,C,H,W = rand0.shape
        zflow = th.zeros((B,T,2,H,W),dtype=rand0.dtype,device=rand0.device)

        # -- run exact --
        _,inds = self.nls(rand0,rand1,zflow,zflow,batchsize)

        # -- compute at randomized inds --
        dists,inds = self.refine(vid0,vid1,inds)

        return dists,inds

def extract_config(cfg):
    cfg = extract_config_nls(cfg)
    cfg = extract_config_refine(cfg)
    return cfg

def init(cfg):
    nls = init_nls(cfg)
    cfg_warnings(cfg)
    refine = init_refine(cfg)
    search = RandIndsSearch(nls,refine)
    return search

def cfg_warnings(cfg):
    pairs = {"wr":1,"kr":-1}
    for key,val in pairs.items():
        if cfg[key] != val:
            cfg[key] = val
            print("WARNING: rand_inds requires (%s,%s). Changing config." % (key,val))
