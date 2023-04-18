#
#
#   API to access the Normalization Methods
#
#


# -- local modules --
from . import softmax

# -- configs --
from dev_basics.configs import ExtractConfig
econfig = ExtractConfig(__file__) # init static variable
extract_config = econfig.extract_config # rename extraction

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Create the initial search function
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@econfig.set_init
def init_normz(cfg):

    # -- unpack --
    econfig.init(cfg)
    cfgs = econfig({"normz":normz_pairs()})
    if econfig.is_init == True: return
    cfg = cfgs.normz

    # -- menu --
    modules = {"softmax":softmax}

    # -- init --
    mod = modules[cfg.normz_name]
    fxn = getattr(mod,'init')(cfg)

    # -- return --
    return fxn

def init(cfg):
    return init_normz(cfg)

def normz_pairs():
    pairs = {"normz_scale":10,
             "normz_name":"softmax",
             "k_n":-1,
             "normz_drop_rate":0.,
             "dist_type":"l2"}
    return pairs

