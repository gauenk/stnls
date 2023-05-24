import copy
dcopy = copy.deepcopy
from easydict import EasyDict as edict

def extract_pairs(_cfg,defaults,restrict=True):
    if restrict: cfg = dcopy(_cfg)
    else: cfg = {}
    for key in defaults:
        if key in cfg and restrict:
            cfg[key] = _cfg[key]
        elif not(key in cfg):
            cfg[key] = defaults[key]
    return edict(cfg)

