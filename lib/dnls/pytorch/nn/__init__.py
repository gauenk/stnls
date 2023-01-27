from . import pfc
# from . import optical_flow_accumulate as ofa
from . import temporal_inds as temporal_inds_f
from . import interpolate_inds as interpolate_inds_f
from . import topk as topk_f
from . import anchor_self as anchor_self_f
from . import jitter_unique_inds as jitter_unique_inds_f

# -- allow to be run dnls.nn.NAME_HERE --
temporal_inds = temporal_inds_f.run
interpolate_inds = interpolate_inds_f.run
topk = topk_f.run
anchor_self = anchor_self_f.run
jitter_unique_inds =jitter_unique_inds_f.run


# -- api v2 --
def init(version,*args,**kwargs):
    if version == "pfc":
        return pfc.PatchFC(*args,**kwargs)
    elif version in ["ofa","optical_flow_accumulate"]:
        return optical_flow_accumulate_f.init(*args)
    elif version == "interpolate_inds":
        return interpolate_inds_f.init(*args)
    else:
        raise ValueError(f"Uknown version [{version}]")

def run(version,*args,**kwargs):
    if version in ["ofa","optical_flow_accumulate"]:
        return optical_flow_accumulate_f.run(*args)
    elif version == "interp_inds":
        return interpolate_inds_f.run(*args)
    elif version in ["topk"]:
        return topk_f.run(*args)
    elif version in ["anchor_self"]:
        return anchor_self_f.run(*args)
    else:
        raise ValueError(f"Uknown version [{version}]")
