"""

Programmtically acesss search functions uniformly

cfg = <pydict of params>
search = stnls.search.init(cfg)

Keys:
search_name: Choose which search function

"""

from . import non_local_stack as non_local_stack_f
from .utils import extract_pairs

# -- easy access --
import importlib,copy
dcopy = copy.deepcopy
from pathlib import Path
from easydict import EasyDict as edict

# # -- configs --
# from dev_basics.configs import ExtractConfig
# econfig = ExtractConfig(__file__) # init static variable
# extract_config = econfig.extract_config # rename extraction

MENU = edict({"nlstack":"non_local_stack"})

def from_search_menu(name):
    if name in MENU:
        return MENU[name]
    else:
        return name

def extract_config(_cfg,restrict=True):
    _cfg = dcopy(_cfg)
    pairs = {"tile_name":"nlstack"}
    search_name = extract_pairs(pairs,_cfg)["tile_name"]
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
    init_s = importlib.import_module("stnls.pytorch.tile.%s" % pkg_name).init
    return init_s(cfg)
