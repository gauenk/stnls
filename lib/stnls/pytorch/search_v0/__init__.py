
# -- packages --
from . import non_local_search as non_local_search_f
from . import non_local_search_pdb as non_local_search_pdb_f
from . import refinement as refinement_f
from . import paired_search as paired_f
from . import approx_space as approx_space_f
from . import approx_time as approx_time_f
from . import approx_spacetime as approx_spacetime_f
from . import window_search as window_search_f
from . import nls_accumulated_flows as nls_accumulated_flows_f
from . import quadref as quadref_f
from . import n3mm_search as n3mm_search_f
from . import nls_pair as nls_pair_f
from .utils import empty_flow,search_wrap

# -- api to programatically access search functions --
from . import api # access uniformly
from .api import init,extract_config

# -- functional api --
nls = non_local_search_f._apply
nls_pdb = non_local_search_pdb_f._apply
refine = refinement_f._apply
paired = paired_f._apply
approx_time = approx_time_f._apply
approx_space = approx_space_f._apply
approx_spacetime = approx_spacetime_f._apply
window = window_search_f._apply
nls_af = nls_accumulated_flows_f._apply
n3mm = n3mm_search_f._apply
nls_pair = nls_pair_f._apply

# -- class api --
NonLocalSearch = non_local_search_f.NonLocalSearch
NonLocalSearchPdb = non_local_search_pdb_f.NonLocalSearchPdb
RefineSearch = refinement_f.RefineSearch
PairedSearch = paired_f.PairedSearch
ApproxTimeSearch = approx_time_f.ApproxTimeSearch
ApproxSpaceSearch = approx_space_f.ApproxSpaceSearch
ApproxSpaceTimeSearch = approx_spacetime_f.ApproxSpaceTimeSearch
WindowSearch = window_search_f.WindowSearch
AccFlowsSearch = nls_accumulated_flows_f.AccFlowsSearch
N3MatMultSearch = n3mm_search_f.N3MatMultSearch
NLSPair = nls_pair_f.NLSPair

