
# -- imports --
from . import topk as topk_f
from . import anchor_self as anchor_self_f
from . import non_local_inds as non_local_inds_f
from . import accumulate_flow as accumulate_flow_f
from . import search_flow as search_flow_f
from . import non_local_attn as non_local_attn_f
from . import non_local_attn_stack as non_local_attn_stack_f
from .accumulate_flow import index_grid

# -- [register] so we can run stnls.nn.NAME_HERE --
topk = topk_f.run
topk_each = topk_f.run_each
anchor_self = anchor_self_f.run
anchor_self_time = anchor_self_f.run_time
anchor_self_refine = anchor_self_f.run_refine
non_local_inds = non_local_inds_f.run
accumulate_flow = accumulate_flow_f.run
extract_search_from_accumulated = accumulate_flow_f.extract_search_from_accumulated
search_flow = search_flow_f.run
NonLocalAttention = non_local_attn_f.NonLocalAttention
NonLocalAttentionStack = non_local_attn_stack_f.NonLocalAttentionStack

