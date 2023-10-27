import copy
dcopy = copy.deepcopy
from easydict import EasyDict as edict

def extract_pairs(_cfg,defaults,restrict=True):
    """

    restrict: if True, only the keys from the "_cfg" are extracted; not filled of default
              if False, the _cfg is copied and missing keys are filled in

    """
    if not(restrict): cfg = dcopy(_cfg)
    else: cfg = {}
    for key in defaults:
        if key in _cfg:
            cfg[key] = _cfg[key]
        elif not(restrict):
            cfg[key] = defaults[key]
        # if (key in cfg) and not(restrict):
        #     cfg[key] = _cfg[key]
        # elif not(key in cfg):
        #     cfg[key] = defaults[key]
    return edict(cfg)

