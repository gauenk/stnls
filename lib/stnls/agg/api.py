"""

Programmtically acesss reducer functions uniformly

cfg = <pydict of params>
reducer = stnls.reducer.init(cfg)

Keys:
agg_name: Choose which aggregate function

"""


# -- easy access --
import importlib
from pathlib import Path
from easydict import EasyDict as edict
from stnls.utils import extract_pairs

MENU = edict({"wpsum":"wpsum",
              "nlstack":"gather",
              "nlgather":"gather",
              "gather":"gather"})

def from_agg_menu(name):
    if name in MENU:
        return MENU[name]
    else:
        return name

def extract_config(_cfg,restrict=True):
    pairs = {"agg_name":"wpsum"}
    agg_name = extract_pairs(_cfg,pairs,restrict=False)["agg_name"]
    pkg_name = from_agg_menu(agg_name)
    base_name = ".".join(__name__.split(".")[:-1])
    mname = "%s.%s" % (base_name,pkg_name)
    extract_config_s = importlib.import_module(mname).extract_config
    cfg = extract_config_s(_cfg)
    cfg.agg_name = agg_name
    return cfg

def init(cfg):
    cfg = extract_config(cfg)
    pkg_name = from_agg_menu(cfg.agg_name)
    init_s = importlib.import_module("stnls.agg.%s" % pkg_name).init
    return init_s(cfg)
