
# -- linalg helpers --
import torch as th # for base cpp
import numpy as np # help
from functools import partial

# -- base cpp --
import dnls_cuda

# -- linking --
from ..link import xla_utils,primitive_utils
xla_shape_to_abstract = xla_utils.xla_shape_to_abstract

# -- jax --
from jax import lax
from jax import dtypes
from jax.abstract_arrays import ShapedArray
from jax._src import abstract_arrays
from jax.lib import xla_client
import jax.numpy as jnp
from jax.interpreters import ad

# -- helpers --
xops = xla_client.ops
_primitive_forward = None
_primitive_backward = None
__all__ = ["run_fwd","run_bwd","init"]

# -=-=-=-=-=-=-=-=-=-=-=-=-
#
#          API
#
# -=-=-=-=-=-=-=-=-=-=-=-=-

def init_fwd(nframes,nqueries,ws_h,ws_w,wt,
             k, ps, pt, chnls,
             stride0, stride1, dilation,
             use_search_abs, reflect_bounds, use_adj,
             oh0, ow0, oh1, ow1, remove_self, full_ws, nbwd,
             use_rand, exact):
    st = 2*wt+1
    dists = jnp.zeros((nqueries,st,ws_h,ws_w),dtype=np.float32)
    inds = jnp.zeros((nqueries,st,ws_h,ws_w,3),dtype=np.int32)
    tranges = jnp.zeros((nframes,nframes),dtype=np.int32)
    n_tranges = jnp.ones((nframes),dtype=np.int32)
    min_tranges = jnp.zeros((nframes),dtype=np.int32)

    # -- init shapes --
    ishapes = jnp.array([0, nqueries,ws_h,ws_w,wt,
                         k, ps, pt, chnls,
                         stride0, stride1, dilation,
                         use_search_abs, reflect_bounds, use_adj,
                         oh0, ow0, oh1, ow1, remove_self, full_ws, nbwd,
                         use_rand, exact, nframes, 0, 0, 0],dtype=jnp.int32)

    run_fwd_part = partial(run_fwd,dists,inds,tranges,n_tranges,min_tranges,ishapes)
    return run_fwd_part

def run_fwd(dists,inds,
            tranges, n_tranges, min_tranges, ishapes,
            vid0, vid1, fflow, bflow, qstart):


    # -- fill shapes --
    nframes,nchnls,height,width = vid0.shape
    ishapes = ishapes.at[-3].set(nchnls)
    ishapes = ishapes.at[-2].set(height)
    ishapes = ishapes.at[-1].set(width)
    ishapes = ishapes.at[0].set(qstart)
    print(ishapes)

    # -- exec cpp primitive --
    out_dists,out_inds = _primitive_forward.bind(
        dists,inds,tranges,n_tranges,min_tranges,ishapes,
        vid0, vid1, fflow, bflow)

    return out_dists,out_inds

def run_bwd(*args,**kwargs):
    return _primitive_backward.bind(*args)

def init(*args,**kwargs): # a "partial" version of run.
    return run_fwd(*args,**kwargs)

def create_opaque_param(*args):
    nelem = len(args)
    pstr= ("%d" * nelem).format(*args)
    return pstr.encode("ascii")

# -=-=-=-=-=-=-=-=-=-=-=-=-
#
#      Jax Primitives
#
# -=-=-=-=-=-=-=-=-=-=-=-=-

def abstract_forward(dists,inds,tranges,n_tranges,
                     min_tranges,ishapes,
                     vid0, vid1, fflow, bflow):
    f32 = dtypes.canonicalize_dtype(np.float32)
    i32 = dtypes.canonicalize_dtype(np.int32)
    return (ShapedArray(dists.shape,f32),ShapedArray(inds.shape,i32))

def forward(xla_builder, dists, inds,
            tranges, n_tranges, min_tranges, ishapes,
            vid0, vid1, fflow, bflow, name=""):

    # -- allocate --
    args = [vid0,vid1,fflow,bflow,tranges,n_tranges,min_tranges,ishapes]

    # -- input --
    input_shapes = [xla_builder.get_shape(arg) for arg in args]
    input_dtypes = tuple(shape.element_type() for shape in input_shapes)
    input_dimensions = tuple(shape.dimensions() for shape in input_shapes)

    # -- output --
    out_shapes = xla_builder.get_shape(dists),xla_builder.get_shape(inds)
    output_shapes = xla_client.Shape.tuple_shape(out_shapes)

    # -- bytes --
    name = name.encode("ascii")
    # oargs = [ps,pt,ws_h,ws_w,wt,chnls,stride0,stride1,
    #          dilation,use_search_abs,reflect_bounds,
    #          use_adj, oh0, ow0, oh1, ow1, remove_self,
    #          full_ws, nbwd, use_rand, exact]
    opaque = b'..'#create_opaque_param(*oargs)

    # -- fill ishapes --
    # ishapes[0] = nframes
    # ishapes[1] = nframes
    # ishapes[2] = nframes
    # ishapes[3] = nframes

    # -- view --
    print("yo.")
    print(input_shapes)
    print(output_shapes)
    print(name)
    print(opaque)

    # -- exec cpp --
    out = xops.CustomCallWithLayout(
        xla_builder,
        name,
        operands=args,
        operand_shapes_with_layout=input_shapes,
        shape_with_layout=output_shapes,
        opaque=opaque
    )
    print(out)
    return out

def abstract_backward(*args):
    print("absbwd: hey.")
    f32 = dtypes.canonicalize_dtype(np.float32)
    return (ShapedArray((3,3,128,128),f32),ShapedArray((3,3,128,128),f32))

def backward(xla_builder,*args,name=""):

    # -- input --
    print("bwd: hey.")
    input_shapes = [xla_builder.get_shape(arg) for arg in args]
    input_dtypes = tuple(shape.element_type() for shape in input_shapes)
    input_dimensions = tuple(shape.dimensions() for shape in input_shapes)

    # -- output --
    output_abstract_arrays = abstract_backward(
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

    print(input_shapes)
    name = name.encode("ascii")
    print(name)
    opaque = b'..'

    # -- exec cpp --
    xops.CustomCallWithLayout(
        xla_builder,
        name,
        operands=args,
        operand_shapes_with_layout=input_shapes,
        shape_with_layout=xla_output_shape,
        opaque=opaque
    )

    # -- output --
    # vid0_grad = jnp.zeros((3,3,128,128),jnp.float32)
    # vid1_grad = jnp.zeros((3,3,128,128),jnp.float32)
    # outs = [lax.zeros_like_array(arg) for arg in args]

    return vid0_grad,vid1_grad

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#      JVP/VJP Use Primitives
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def backward_jvp(arg_values, arg_tangents, name_fwd="", name_bwd=""):

    # -- primal --
    print("a: ")
    print(len(arg_values))
    print("b: ")
    print(len(arg_tangents))
    fwd_out = run_fwd(*arg_values)

    # -- tangent --
    def make_zero(tan):
        return lax.zeros_like_array(x) if type(tan) is ad.Zero else tan
    bwd_out = run_bwd(*arg_values) # output_tangent

    return (fwd_out, bwd_out)

def backward_vjp(*args,name_fwd="",name_bwd=""): # "backward"

    print("a: ")
    print(len(args))

    # -- primal --
    fwd_out = run_fwd(*arg_values)

    # -- tangent --
    def make_zero(tan):
        return lax.zeros_like_array(x) if type(tan) is ad.Zero else tan
    bwd_out = run_bwd(*arg_values) # output_tangent
    return (fwd_out, bwd_out)


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Jax Registration
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def _register():

    # -- assign primitive for api --
    global _primitive_forward
    global _primitive_backward

    # -- unpack c++ --
    name = "search_prod_with_index_jax"
    pair = dnls_cuda.search_prod_with_index_jax()
    fwd_cpp,bwd_cpp = pair['forward'],pair['backward']

    # -- register c++ --
    name_fwd,name_bwd = primitive_utils.xla_register(name, fwd_cpp, bwd_cpp)

    # -- wrap --
    fwd = partial(forward,name=name_fwd)
    bwd = partial(backward,name=name_bwd)
    jvp = partial(backward_jvp,name_fwd=name_fwd,name_bwd=name_bwd)
    vjp = partial(backward_vjp,name_fwd=name_fwd,name_bwd=name_bwd)

    # -- define primitive --
    name = "search_prod_with_index_jax_forward"
    prim_fwd = primitive_utils.cfunc_to_jax(name,fwd,abstract_forward,batching_fn=None)
    name = "search_prod_with_index_jax_backward"
    prim_bwd = primitive_utils.cfunc_to_jax(name,bwd,abstract_backward,batching_fn=None)

    # -- assign primitive for api --
    _primitive_forward = prim_fwd
    _primitive_backward = prim_bwd

    # -- assign jvp/vjp for forward --
    ad.primitive_jvps[prim_fwd] = jvp
    # ad.primitive_transposes[prim_fwd] = vjp

_register() # call for init
