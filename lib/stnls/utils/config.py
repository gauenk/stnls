import copy
dcopy = copy.deepcopy

def extract_pairs(_cfg,defaults,restrict=True):
    if only_included: cfg = dcopy(_cfg)
    else: cfg = {}
    for key in defaults:
        if key in cfg and restrict:
            cfg[key] = _cfg[key]
        elif not(key in cfg):
            cfg[key] = defaults[key]
    return cfg
            
