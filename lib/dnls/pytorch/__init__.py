
# -- submodules --
from . import search
from . import reducers
from . import tile
from . import tile_k
from . import simple
from . import testing

#
# -- unpack functions into namespace --
#

# -- tiling --
from .tile.fold import fold,Fold
from .tile.unfold import unfold,Unfold
from .tile.ifold import ifold,iFold
from .tile.ifoldz import ifoldz,iFoldz
from .tile.iunfold import iunfold,iUnfold

# -- tiling k --
from .tile_k.fold_k import fold_k,FoldK
from .tile_k.unfold_k import unfold_k,UnfoldK

