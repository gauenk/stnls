#
#
#   API to access the Aggregation Methods
#
#


# -- local modules --
from . import wpsum
from . import pdbsum

# -- configs --
from stnls.utils import config

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Create the initial search function
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_config(cfg,restrict=True):
    cfg = config.extract_pairs(cfg,default_pairs(),restrict=restrict)
    return cfg

def init_agg(cfg):

    # -- unpack --
    cfg = extract_config(cfg)

    # -- menu --
    modules = {"wpsum":wpsum,"pdbsum":pdbsum}

    # -- init --
    mod = modules[cfg.agg_name]
    fxn = getattr(mod,'init')(cfg)

    # -- return --
    return fxn

def init(cfg):
    return init_agg(cfg)

def default_pairs():
    pairs = {"ps":7,"pt":1,"dilation":1,
             "exact":False,"reflect_bounds":False,
             "k_a":-1,"agg_name":"wpsum",
             "stride0":4,"pdbagg_chunk_size":512}
    return pairs
