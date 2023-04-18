"""

Programmtically acesss search functions uniformly

cfg = <pydict of params>
search = stnls.search.init(cfg)

Keys:
search_name: Choose which search function

"""

from . import non_local_search
from . import refinement
from . import approx_space
from . import approx_time
from . import approx_spacetime
from . import window_search
from . import nls_accumulated_flows
from . import rand_inds
from .utils import extract_pairs


# -- easy access --
import importlib
from pathlib import Path
from easydict import EasyDict as edict

# # -- configs --
# from dev_basics.configs import ExtractConfig
# econfig = ExtractConfig(__file__) # init static variable
# extract_config = econfig.extract_config # rename extraction

MENU = edict({"exact":"non_local_search",
              "nls":"non_local_search",
              "nl":"non_local_search",
              "refine":"refinement",
              "approx_t":"approx_time",
              "nlat":"approx_time",
              "approx_s":"approx_space",
              "nlas":"approx_space",
              "approx_st":"approx_spacetime",
              "nlast":"approx_spacetime",
              "rand_inds":"rand_inds"})

def from_search_menu(name):
    if name in MENU:
        return MENU[name]
    else:
        return name

def extract_config(_cfg):
    pairs = {"search_name":"nls"}
    search_name = extract_pairs(pairs,_cfg)["search_name"]
    pkg_name = from_search_menu(search_name)
    base_name = ".".join(__name__.split(".")[:-1])
    mname = "%s.%s" % (base_name,pkg_name)
    extract_config_s = importlib.import_module(mname).extract_config
    cfg = extract_config_s(_cfg)
    cfg.search_name = search_name
    return cfg

def init(cfg):
    cfg = extract_config(cfg)
    pkg_name = from_search_menu(cfg.search_name)
    init_s = importlib.import_module("stnls.pytorch.search.%s" % pkg_name).init
    return init_s(cfg)
