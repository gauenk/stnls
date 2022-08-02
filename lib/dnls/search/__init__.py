from . import l2_search
from . import l2_search_with_index
from . import prod_search
from . import prod_search_with_index


def init(version,*args,**kwargs):
    if version == "l2":
        return l2_search.L2Search(*args,**kwargs)
    elif version == "l2_with_index":
        return l2_search_with_index.L2Search_with_index(*args,**kwargs)
    elif version == "prod":
        return prod_search.ProductSearch(*args,**kwargs)
    elif version == "prod_with_index":
        return prod_search_with_index.ProductSearch_with_index(*args,**kwargs)
    else:
        raise ValueError(f"Uknown version [{version}]")
