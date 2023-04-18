import sys, os
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)

# -- api --
from . import pytorch
# from . import jax # jax after pytorch
from . import utils
from . import flow
from . import dev

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Init submodules for Pytorch
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# -- submodules --
from .pytorch import nn
from .pytorch import search
from .pytorch import normz
from .pytorch import agg
from .pytorch import reducers
from .pytorch import tile
from .pytorch import tile_k
from .pytorch import simple
from .pytorch import testing
from .pytorch import warp
from .dev import search as search_dev

#
# -- unpack functions into namespace --
#

# -- tiling --
from .pytorch.tile.fold import fold,Fold
from .pytorch.tile.unfold import unfold,Unfold
from .pytorch.tile.ifold import ifold,iFold
from .pytorch.tile.ifoldz import ifoldz,iFoldz
from .pytorch.tile.iunfold import iunfold,iUnfold

# -- tiling k --
from .pytorch.tile_k.fold_k import fold_k,FoldK
from .pytorch.tile_k.unfold_k import unfold_k,UnfoldK

