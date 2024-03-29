"""

Programmtically acesss search functions uniformly

cfg = <pydict of params>
search = stnls.search.init(cfg)

Keys:
search_name: Choose which search function

"""

from . import non_local_search
from . import refinement
from . import paired_search
from . import rand_inds
from . import n3mm_search
from stnls.utils import extract_pairs

# -- easy access --
import importlib,copy
dcopy = copy.deepcopy
from pathlib import Path
from easydict import EasyDict as edict


MENU = edict({"exact":"non_local_search",
              "nls":"non_local_search",
              "nl":"non_local_search",
              "refine":"refinement",
              "pair":"paired_search",
              "paired":"paired_search",
              "paired_refine":"paired_refine",
              "paired_ref":"paired_refine",
              "rand_inds":"rand_inds",
              "n3mm":"n3mm_search"})

def from_search_menu(name):
    if name in MENU:
        return MENU[name]
    else:
        return name

def extract_config(_cfg,restrict=True):
    _cfg = dcopy(_cfg)
    pairs = {"search_name":"nls"}
    search_name = extract_pairs(_cfg,pairs,restrict=False)["search_name"]
    pkg_name = from_search_menu(search_name)
    base_name = ".".join(__name__.split(".")[:-1])
    mname = "%s.%s" % (base_name,pkg_name)
    extract_config_s = importlib.import_module(mname).extract_config
    cfg = extract_config_s(_cfg)
    cfg.search_name = search_name
    return cfg

def init(cfg):
    cfg = extract_config(cfg,False)
    pkg_name = from_search_menu(cfg.search_name)
    init_s = importlib.import_module("stnls.search.%s" % pkg_name).init
    return init_s(cfg)
