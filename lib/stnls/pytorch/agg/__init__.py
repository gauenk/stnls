"""

Methos for aggregating non-local patches

"""

from . import api
# from . import compare
from .api import init,init_agg,extract_config

from .pdbsum import PdbAgg
from .wpsum import WeightedSum

