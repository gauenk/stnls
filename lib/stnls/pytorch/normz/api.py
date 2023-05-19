#
#
#   API to access the Normalization Methods
#
#


# -- local modules --
from . import softmax

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

def init_normz(cfg):

    # -- unpack --
    cfg = extract_config(cfg)

    # -- menu --
    modules = {"softmax":softmax}

    # -- init --
    mod = modules[cfg.normz_name]
    fxn = getattr(mod,'init')(cfg)

    # -- return --
    return fxn

def init(cfg):
    return init_normz(cfg)

def default_pairs():
    pairs = {"normz_scale":10,
             "normz_name":"softmax",
             "k_n":-1,
             "normz_drop_rate":0.,
             "dist_type":"l2"}
    return pairs

