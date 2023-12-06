
# -- modules --
from . import pool as pool_f
from . import gather as gather_f
from . import gather_add as gather_add_f
from . import scatter as scatter_f
from . import scatter_add as scatter_add_f
from . import scatter_labels as scatter_labels_f
from . import scatter_tensor as scatter_tensor_f

# -- api to programatically access search functions --
from . import api # access uniformly
from .api import init,extract_config

# -- functional api --
pool = pool_f._apply
gather = gather_f._apply
gather_add = gather_add_f._apply
scatter = scatter_f._apply
scatter_add = scatter_add_f._apply
scatter_labels = scatter_labels_f.run
scatter_tensor = scatter_tensor_f.run
scatter_topk = scatter_tensor_f.run_topk

# -- class api --
NonLocalGather = gather_f.NonLocalGather
NonLocalScatter = scatter_f.NonLocalScatter
NonLocalGatherAdd = gather_add_f.NonLocalGatherAdd
NonLocalScatterAdd = scatter_add_f.NonLocalScatterAdd
PooledPatchSum = pool_f.PooledPatchSum
