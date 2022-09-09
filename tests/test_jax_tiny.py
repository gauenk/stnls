import dnls
import dnls_cuda
from functools import partial
from jax.lib import xla_client
from jax import core, dtypes, lax
from jax.interpreters import ad, batching, xla
from jax.abstract_arrays import ShapedArray
xla_client.register_custom_call_target('example',dnls_cuda.reg()['example'],"gpu")

# src/kepler_jax/kepler_jax.py
import numpy as np

def xla_shape_to_abstract(xla_shape) -> ShapedArray:
    """
    Converts an XLA shape to a Jax ShapedArray object, which
    is the empty shell defining only shape and dtype used by
    abstract evaluation
    """
    return ShapedArray(xla_shape.dimensions(), xla_shape.element_type())


def create_xla_target_capsule(ptr):
    """
    Wraps a C function pointer into an XLA-compatible PyCapsule.
    Assumes that the function pointed at by the pointer `ptr`
    respects the XLA calling convention (3 void* arguments).
    """
    # Magic name that the PyCapsule must have to be recognized
    # by XLA as a custom call
    xla_capsule_magic = b"xla._CUSTOM_CALL_TARGET"

    return pycapsule_new(ptr, xla_capsule_magic)

def _simple_function(xla_builder,vid0,vid1,fflow,bflow,dists,inds,
                     tranges,n_tranges,min_tranges,abstract_eval_fn=None):
    # -- args --
    args = [vid0,vid1,fflow,bflow,dists,inds,tranges,n_tranges,min_tranges]

    # -- input --
    input_shapes = [xla_builder.get_shape(arg) for arg in args]
    input_dtypes = tuple(shape.element_type() for shape in input_shapes)
    input_dimensions = tuple(shape.dimensions() for shape in input_shapes)

    # -- output --
    output_abstract_arrays = abstract_eval_fn(
        *[xla_shape_to_abstract(shape) for shape in input_shapes]
    )
    output_shapes = tuple(array.shape for array in output_abstract_arrays)
    output_dtypes = tuple(array.dtype for array in output_abstract_arrays)
    output_layouts = map(lambda shape: range(len(shape) - 1, -1, -1), output_shapes)
    xla_output_shapes = [
        xla_client.Shape.array_shape(*arg)
        for arg in zip(output_dtypes, output_shapes, output_layouts)
    ]
    xla_output_shape = xla_client.Shape.tuple_shape(xla_output_shapes)

    # We dispatch a different call depending on the dtype
    op_name = b"example"

    # On the CPU, we pass the size of the data as a the first input
    # argument
    opaque = b".."

    return xla_client.ops.CustomCallWithLayout(
        xla_builder,
        op_name,
        operands=args,
        operand_shapes_with_layout=input_shapes,
        shape_with_layout=xla_output_shape,
        opaque=opaque
    )

def _simple_abstract(*args):
    # shape = vid0.shape
    # dtype = dtypes.canonicalize_dtype(vid0.dtype)
    # assert dtypes.canonicalize_dtype(vid0.dtype) == dtype
    # assert vid0.shape == shape
    #[xla_shape_to_abstract(elem) for elem in args]
    return [ShapedArray(args[0].shape,args[0].dtype)]

# -- creat primitive --
_my_prim = core.Primitive("example")
_my_prim.multiple_results = True
_my_prim.def_impl(partial(xla.apply_primitive, _my_prim))
_my_prim.def_abstract_eval(_simple_abstract)
xla.backend_specific_translations["gpu"][_my_prim] = partial(_simple_function,abstract_eval_fn=_simple_abstract)

# Connect the JVP and batching rules
# ad.primitive_jvps[_my_prim] = _kepler_jvp
# batching.primitive_batchers[_my_prim] = _kepler_batch


def test_my_prim():
    vid0 = np.zeros((3,3,128,128),dtype=np.float32)
    vid1 = np.zeros((3,3,128,128),dtype=np.float32)
    fflow = np.zeros((3,2,128,128),dtype=np.float32)
    bflow = np.zeros((3,2,128,128),dtype=np.float32)
    dists = np.zeros((1,1,1,1),dtype=np.float32)
    inds = np.zeros((1,1,1,1,3),dtype=np.int32)
    tranges = np.zeros((3,3),dtype=np.int32)
    n_tranges = np.zeros((3),dtype=np.int32)
    min_tranges = np.zeros((3),dtype=np.int32)
    # vid0,vid1,fflow,bflow,dists,inds,
    # qstart, stride0, n_h0, n_w0,
    # ps,pt,ws_h,ws_w,wt,chnls,stride,dilation,
    # use_search_abs, use_bounds, use_adj,
    # full_ws, oh0, ow0, oh1, ow1,
    # tranges, n_tranges, min_tranges);
    _my_prim.bind(vid0,vid1,fflow,bflow,dists,inds,
                  tranges,n_tranges,min_tranges)
if __name__ == "__main__":
    test_my_prim()
