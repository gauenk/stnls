#
#
# -- API Utils --
#
#


def extract_pairs(pairs,_cfg):
    cfg = edict()
    for key,default in pairs.items():
        if key in _cfg:
            cfg[key] = _cfg[key]
        else:
            cfg[key] = pairs[key]
    return cfg
