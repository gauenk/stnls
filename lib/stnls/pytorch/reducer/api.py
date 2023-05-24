"""

Programmtically acesss reducer functions uniformly

cfg = <pydict of params>
reducer = stnls.reducer.init(cfg)

Keys:
reducer_name: Choose which reducer function

"""


# -- easy access --
import importlib
from pathlib import Path
from easydict import EasyDict as edict
from stnls.utils import extract_pairs

MENU = edict({"wpsum":"wpsum",
              "iwpsum":"iwpsum",
              "fwpsum":"fwpsum"})

def from_reducer_menu(name):
    if name in MENU:
        return MENU[name]
    else:
        return name

def extract_config(_cfg):
    pairs = {"reducer_name":"iwpsum"}
    reducer_name = extract_pairs(_cfg,pairs)["reducer_name"]
    pkg_name = from_reducer_menu(reducer_name)
    base_name = ".".join(__name__.split(".")[:-1])
    mname = "%s.%s" % (base_name,pkg_name)
    extract_config_s = importlib.import_module(mname).extract_config
    cfg = extract_config_s(_cfg)
    cfg.reducer_name = reducer_name
    return cfg

def init(cfg):
    cfg = extract_config(cfg)
    pkg_name = from_reducer_menu(cfg.reducer_name)
    init_s = importlib.import_module("stnls.pytorch.reducer.%s" % pkg_name).init
    return init_s(cfg)
