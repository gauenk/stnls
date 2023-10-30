
# -- packages --
from . import non_local_search as non_local_search_f
from . import refinement as refinement_f
from . import paired_search as paired_search_f
from . import paired_refine as paired_refine_f
from . import n3mm_search as n3mm_search_f
from .utils import empty_flow,search_wrap
from .utils import get_time_window_inds
# from .paired_utils import get_time_window_inds

# -- api to programatically access search functions --
from . import api # access uniformly
from .api import init,extract_config

# -- functional api --
nls = non_local_search_f._apply
refine = refinement_f._apply
paired_search = paired_search_f._apply
paired_refine = paired_refine_f._apply
n3mm = n3mm_search_f._apply

# -- class api --
NonLocalSearch = non_local_search_f.NonLocalSearch
RefineSearch = refinement_f.RefineSearch
PairedSearch = paired_search_f.PairedSearch
PairedRefine = paired_refine_f.PairedRefine
N3MatMultSearch = n3mm_search_f.N3MatMultSearch

