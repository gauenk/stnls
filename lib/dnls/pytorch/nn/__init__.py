
# -- imports --
from . import pfc
from . import topk as topk_f
from . import anchor_self as anchor_self_f
from . import temporal_inds as temporal_inds_f
from . import accumulate_flow as accumulate_flow_f
from . import interpolate_inds as interpolate_inds_f
from . import jitter_unique_inds as jitter_unique_inds_f
from . import compare_inds as compare_inds_f
from . import flow_patches as flow_patches_f

# -- [register] so we can run dnls.nn.NAME_HERE --
topk = topk_f.run
anchor_self = anchor_self_f.run
temporal_inds = temporal_inds_f.run
accumulate_flow = accumulate_flow_f.run
interpolate_inds = interpolate_inds_f.run
jitter_unique_inds = jitter_unique_inds_f.run
compare_inds = compare_inds_f.run
flow_patches = flow_patches_f.get_patches
flow_patches_mse = flow_patches_f.get_mse

# -- api v2 --
def init(version,*args,**kwargs):
    if version == "pfc":
        return pfc.PatchFC(*args,**kwargs)
    elif version in ["accumulate_flow"]:
        return accumulate_flow_f.init(*args)
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
