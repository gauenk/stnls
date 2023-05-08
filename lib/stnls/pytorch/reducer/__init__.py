

# -- modules --
from . import wpsum as wpsum_f
from . import iwpsum as iwpsum_f
from . import fwpsum as fwpsum_f

# -- api to programatically access search functions --
from . import api # access uniformly
from .api import init,extract_config

# -- functional api --
wpsum = wpsum_f._apply
iwpsum = iwpsum_f._apply

# -- class api --
WeightedPatchSum = wpsum_f.WeightedPatchSum
FoldedWeightedPatchSum = fwpsum_f.FoldedWeightedPatchSum
InplaceWeightedPatchSum = iwpsum_f.InplaceWeightedPatchSum
