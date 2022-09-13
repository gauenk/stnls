import dnls
import dnls_cuda
from functools import partial
import jax.numpy as jnp
from jax.lib import xla_client
from jax import core, dtypes, lax
from jax.interpreters import ad, batching, xla
from jax.abstract_arrays import ShapedArray
import numpy as np

# "search_prod_with_jax"
# name = "prod_search_with_index"
# xla_client.register_custom_call_target('prod_search_with_index_forward',
#                                        dnls_cuda.search_prod_with_jax()['forward'],
#                                        "gpu")

# # src/kepler_jax/kepler_jax.py
# import numpy as np

# import numba as nb
# from numba import types as nb_types

# -- init args --
vid0 = np.random.rand(3,3,128,128).astype(np.float32)
vid1 = np.random.rand(3,3,128,128).astype(np.float32)
fflow = np.zeros((3,2,128,128),dtype=np.float32)
bflow = np.zeros((3,2,128,128),dtype=np.float32)
nframes = vid0.shape[0]
qstart, nqueries = 0,10
k, ps, pt = 5, 7, 1
ws_h, ws_w, wt, chnls = 5, 5, 0, -1
stride0, stride1, dilation = 1, 1, 1
use_search_abs, reflect_bounds = False, True
use_adj, use_k = True, True
oh0, ow0, oh1, ow1 = 0, 0, 0, 0
remove_self, full_ws, nbwd = False, True, 1
use_rand, exact = False, False

args = [0,vid0,vid1,fflow,bflow]

from jax._src import api
import dnls
print(dnls.jax.search.prod_search_with_index._register())
fxn = dnls.jax.search.prod_search_with_index.run_fwd
# print(dnls.jax.search.prod_search_with_index.forward())
# iargs = [nframes,nqueries,ws_h,ws_w,wt,
#          k, ps, pt, chnls, stride0, stride1, dilation,
#          use_search_abs, reflect_bounds, use_adj,
#          oh0, ow0, oh1, ow1, remove_self, full_ws, nbwd,
#          use_rand, exact]
# fxn = dnls.jax.search.prod_search_with_index.init_fwd(*iargs)
# fxn()

# dists,inds = api.jit(fxn,static_argnums=(4,5,6,))(*args)
# print("api.jit(fxn): ",api.jit(fxn))
# dists,inds = fxn(vid0, vid1, fflow, bflow, qstart=qstart,
#                  nqueries=nqueries,ws_h=ws_h,ws_w=ws_w,wt=wt,k=k)
dists,inds = api.jit(fxn,static_argnums=(4,5,6,7,8,9))(
    vid0, vid1, qstart, nqueries, fflow, bflow,
    ws_h=ws_h,ws_w=ws_w,wt=wt,k=k)
print(dists.shape)
print(inds.shape)
exit(0)
# print(dists[0])
# print(dists[1])
# print(inds[0])
# dists = jnp.reshape(dists,(nqueries,-1))
# inds = jnp.reshape(inds,(nqueries,-1,3))
# print(dists[:3,:3])
# print(dists[:,:3])
# print(inds[:3,:3])
# print(inds[:,:3])

# forward
# my_jvp = api.jit(lambda ins,tans: api.jvp(fxn, ins, tans))
# my_jvp(vid0,vid1,fflow,bflow,

print(api.grad(fxn,argnums=1)(vid0,vid1,fflow,bflow))


# def xla_shape_to_abstract(xla_shape) -> ShapedArray:
#     """
#     Converts an XLA shape to a Jax ShapedArray object, which
#     is the empty shell defining only shape and dtype used by
#     abstract evaluation
#     """
#     return ShapedArray(xla_shape.dimensions(), xla_shape.element_type())

# def create_xla_target_capsule(ptr):
#     """
#     Wraps a C function pointer into an XLA-compatible PyCapsule.
#     Assumes that the function pointed at by the pointer `ptr`
#     respects the XLA calling convention (3 void* arguments).
#     """
#     # Magic name that the PyCapsule must have to be recognized
#     # by XLA as a custom call
#     xla_capsule_magic = b"xla._CUSTOM_CALL_TARGET"

#     return pycapsule_new(ptr, xla_capsule_magic)

# def _simple_function(xla_builder,vid0,vid1,fflow,bflow,dists,inds,
#                      tranges,n_tranges,min_tranges,abstract_eval_fn=None):
#     # -- args --
#     args = [vid0,vid1,fflow,bflow,dists,inds,tranges,n_tranges,min_tranges]

#     # -- input --
#     input_shapes = [xla_builder.get_shape(arg) for arg in args]
#     input_dtypes = tuple(shape.element_type() for shape in input_shapes)
#     input_dimensions = tuple(shape.dimensions() for shape in input_shapes)

#     # -- output --
#     output_abstract_arrays = abstract_eval_fn(
#         *[xla_shape_to_abstract(shape) for shape in input_shapes]
#     )
#     output_shapes = tuple(array.shape for array in output_abstract_arrays)
#     output_dtypes = tuple(array.dtype for array in output_abstract_arrays)
#     output_layouts = map(lambda shape: range(len(shape) - 1, -1, -1), output_shapes)
#     xla_output_shapes = [
#         xla_client.Shape.array_shape(*arg)
#         for arg in zip(output_dtypes, output_shapes, output_layouts)
#     ]
#     xla_output_shape = xla_client.Shape.tuple_shape(xla_output_shapes)

#     # We dispatch a different call depending on the dtype
#     op_name = b"example"

#     # On the CPU, we pass the size of the data as a the first input
#     # argument
#     opaque = b".."

#     return xla_client.ops.CustomCallWithLayout(
#         xla_builder,
#         op_name,
#         operands=args,
#         operand_shapes_with_layout=input_shapes,
#         shape_with_layout=xla_output_shape,
#         opaque=opaque
#     )

# def _simple_abstract(*args):
#     # shape = vid0.shape
#     # dtype = dtypes.canonicalize_dtype(vid0.dtype)
#     # assert dtypes.canonicalize_dtype(vid0.dtype) == dtype
#     # assert vid0.shape == shape
#     #[xla_shape_to_abstract(elem) for elem in args]
#     return [ShapedArray(args[0].shape,args[0].dtype)]

# # -- creat primitive --
# _my_prim = core.Primitive("example")
# _my_prim.multiple_results = True
# _my_prim.def_impl(partial(xla.apply_primitive, _my_prim))
# _my_prim.def_abstract_eval(_simple_abstract)
# xla.backend_specific_translations["gpu"][_my_prim] = partial(_simple_function,abstract_eval_fn=_simple_abstract)

# # Connect the JVP and batching rules
# # ad.primitive_jvps[_my_prim] = _kepler_jvp
# # batching.primitive_batchers[_my_prim] = _kepler_batch


# def test_my_prim():
#     vid0 = np.zeros((3,3,128,128),dtype=np.float32)
#     vid1 = np.zeros((3,3,128,128),dtype=np.float32)
#     fflow = np.zeros((3,2,128,128),dtype=np.float32)
#     bflow = np.zeros((3,2,128,128),dtype=np.float32)
#     dists = np.zeros((1,1,1,1),dtype=np.float32)
#     inds = np.zeros((1,1,1,1,3),dtype=np.int32)
#     tranges = np.zeros((3,3),dtype=np.int32)
#     n_tranges = np.zeros((3),dtype=np.int32)
#     min_tranges = np.zeros((3),dtype=np.int32)
#     # vid0,vid1,fflow,bflow,dists,inds,
#     # qstart, stride0, n_h0, n_w0,
#     # ps,pt,ws_h,ws_w,wt,chnls,stride,dilation,
#     # use_search_abs, use_bounds, use_adj,
#     # full_ws, oh0, ow0, oh1, ow1,
#     # tranges, n_tranges, min_tranges);
#     _my_prim.bind(vid0,vid1,fflow,bflow,dists,inds,
#                   tranges,n_tranges,min_tranges)
# if __name__ == "__main__":
#     test_my_prim()
