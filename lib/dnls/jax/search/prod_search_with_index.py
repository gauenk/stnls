
# -- linalg helpers --
import torch as th # for base cpp
import numpy as np # help

# -- base cpp --
import dnls_cuda

# -- linking --
from ..link import xla_utils,primitive_utils
# from ..link import _method

# -- jax --
from jax import dtypes
from jax.abstract_arrays import ShapedArray

_primitive = None

def run(*args,**kwargs):
    return _primitive.bind(args)

def get_run(*args,**kwargs): # a "partial" version of run.
    return run(*args,**kwargs)

def abstract(*args):
    dtype = dtypes.canonicalize_dtype(np.int32)
    return [ShapedArray((0,),dtype)]

def wrap_fwd(fwd):
    def wrap(fwd,*args,**kwargs):
        return fwd(*args,**kwargs)
    return wrap

def wrap_bwd(bwd):
    def wrap(bwd,*args,**kwargs):
        return bwd(*args,**kwargs)
    return wrap


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Jax Registration
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def _register():
    global _primitive
    name = "search_prod_with_index_jax"
    pair = dnls_cuda.search_prod_with_index_jax()
    fwd,bwd = pair['forward'],pair['backward']
    wfwd = wrap_fwd(fwd)
    wbwd = wrap_bwd(bwd)
    prim = primitive_utils.cfunc_to_jax("search_prod_with_index_jax",
                                        fwd,bwd,wfwd,wbwd,abstract,batching_fn=None)
    _primitive = prim
_register()
