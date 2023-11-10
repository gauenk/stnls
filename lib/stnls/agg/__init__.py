
# -- modules --
from . import wpsum as wpsum_f
from . import gather as gather_f
from . import scatter as scatter_f

# -- api to programatically access search functions --
from . import api # access uniformly
from .api import init,extract_config

# -- functional api --
wpsum = wpsum_f._apply
gather = gather_f._apply
scatter = scatter_f._apply

# -- class api --
NonLocalGather = gather_f.NonLocalGather
NonLocalScatter = scatter_f.NonLocalScatter
WeightedPatchSum = wpsum_f.WeightedPatchSum
