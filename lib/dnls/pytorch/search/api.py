"""

Programmtically acesss search functions uniformly

"""

from . import non_local_search
from . import refinement
from . import approx_space
from . import approx_time
from . import approx_spacetime
from . import window_search
from . import nls_accumulated_flows
from .utils import extract_pairs


# -- easy access --
import importlib
from pathlib import Path
from easydict import EasyDict as edict

def search_menu(name):
    menu = edict({"exact":"non_local_search","nls":"non_local_search",
                  "refine":"refinement",
                  "approx_t":"approx_time","approx_s":"approx_space",
                  "approx_st":"approx_spacetime"})
    if name in menu:
        return menu[name]
    else:
        return name

def extract_config(_cfg):
    pairs = {"search_name":"nls"}
    search_name = extract_pairs(pairs,_cfg)["search_name"]
    pkg_name = search_menu(search_name)
    base_name = ".".join(__name__.split(".")[:-1])
    mname = "%s.%s" % (base_name,pkg_name)
    extract_config_s = importlib.import_module(mname).extract_config
    cfg = extract_config_s(_cfg)
    cfg.search_name = search_name
    return cfg

def init(cfg):
    cfg = extract_config(cfg)
    pkg_name = search_menu(cfg.search_name)
    init_s = importlib.import_module("dnls.pytorch.search.%s" % pkg_name).init
    return init_s(cfg)
