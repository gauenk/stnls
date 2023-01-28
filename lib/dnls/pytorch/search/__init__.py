from . import l2_dists
from . import l2_search
from . import l2_search_with_index
from . import l2_search_with_heads
from . import prod_search
from . import prod_search_with_index
from . import prod_pf_search_with_index
from . import prod_search_with_heads
from . import prod_dists
from . import prod_refine
from . import window_search
from . import search_with_heads

def init(version,*args,**kwargs):
    if version == "l2":
        return l2_search.L2Search(*args,**kwargs)
    elif version == "l2_dists":
        return l2_dists.L2Dists(*args,**kwargs)
    elif version == "window":
        return window_search.WindowSearch(*args,**kwargs)
    elif version == "l2_with_index":
        return l2_search_with_index.L2Search_with_index(*args,**kwargs)
    elif version == "l2_with_heads":
        return l2_search_with_heads.L2Search_with_index(*args,**kwargs)
    elif version == "prod":
        return prod_search.ProductSearch(*args,**kwargs)
    elif version == "prod_with_index":
        return prod_search_with_index.ProductSearch_with_index(*args,**kwargs)
    elif version == "prod_pf_with_index":
        return prod_pf_search_with_index.ProductPfSearch_with_index(*args,**kwargs)
    elif version in ["prod_with_heads","prod_search_with_heads"]:
        return prod_search_with_heads.ProdSearchWithHeads(*args,**kwargs)
    elif version == "prod_dists":
        return prod_dists.ProdDistsWithHeads(*args,**kwargs)
    elif version == "prod_refine":
        return prod_refine.ProdRefineWithHeads(*args,**kwargs)
    elif version == "search_with_heads":
        return search_with_heads.SearchWithHeads(*args,**kwargs)
    else:
        raise ValueError(f"Uknown version [{version}]")
