
# -- imports --
from . import pfc
from . import topk as topk_f
from . import topk_time as topk_time_f
from . import anchor_self as anchor_self_f
from . import temporal_inds as temporal_inds_f
from . import non_local_inds as non_local_inds_f
from . import accumulate_flow as accumulate_flow_f
from . import interpolate_inds as interpolate_inds_f
from . import jitter_unique_inds as jitter_unique_inds_f
from . import compare_inds as compare_inds_f
from . import flow_patches as flow_patches_f
from . import non_local_attn as non_local_attn_f
from . import remove_same_frame as remove_same_frame_f

# -- [register] so we can run stnls.nn.NAME_HERE --
topk = topk_f.run
topk_time = topk_time_f.run
anchor_self = anchor_self_f.run
temporal_inds = temporal_inds_f.run
non_local_inds = non_local_inds_f.run
accumulate_flow = accumulate_flow_f.run
interpolate_inds = interpolate_inds_f.run
jitter_unique_inds = jitter_unique_inds_f.run
compare_inds = compare_inds_f.run
flow_patches = flow_patches_f.get_patches
flow_patches_mse = flow_patches_f.get_mse
NonLocalAttention = non_local_attn_f.NonLocalAttention
remove_same_frame = remove_same_frame_f.run

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