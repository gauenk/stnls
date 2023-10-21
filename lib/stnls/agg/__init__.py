

# -- modules --
from . import wpsum as wpsum_f
from . import non_local_stack as nlstack_f

# -- api to programatically access search functions --
from . import api # access uniformly
from .api import init,extract_config

# -- functional api --
# wpsum = wpsum_f._apply
nlstack = nlstack_f._apply

# -- class api --
WeightedPatchSum = wpsum_f.WeightedPatchSum
NonLocalStack = nlstack_f.NonLocalStack
