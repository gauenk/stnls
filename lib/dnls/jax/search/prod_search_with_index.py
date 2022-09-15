3
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
import jax
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

def init(fflow, bflow,
         nframes, k, ps, pt, ws, wt, oh0, ow0, oh1, ow1,
         chnls=-1,dilation=1,stride0=1, stride1=1,
         reflect_bounds=False,use_k=False,
         remove_self=False,full_ws=False,
         search_abs=True,use_adj=False,exact=False,
         rbwd=False,nbwd=1):

    # -- alloc memory --
    # st = 2*wt+1
    # dists = jnp.zeros((nqueries,st,ws_h,ws_w),dtype=np.float32)
    # inds = jnp.zeros((nqueries,st,ws_h,ws_w,3),dtype=np.int32)
    # tranges = jnp.zeros((nframes,nframes),dtype=np.int32)
    # n_tranges = jnp.ones((nframes),dtype=np.int32)
    # min_tranges = jnp.zeros((nframes),dtype=np.int32)

    # -- init shapes --
    # ishapes = jnp.array([0, nqueries,ws_h,ws_w,wt,
    #                      k, ps, pt, chnls,
    #                      stride0, stride1, dilation,
    #                      use_search_abs, reflect_bounds, use_adj,
    #                      oh0, ow0, oh1, ow1, remove_self, full_ws, nbwd,
    #                      rbwd, exact, nframes, 0, 0, 0],dtype=jnp.int32)
    run_fwd_part = partial(run_fwd, fflow=fflow,bflow=bflow,
                           k=k, ps=ps, pt=pt, chnls=chnls,
                           stride0=stride0, stride1=stride1, dilation=dilation,
                           ws_h=ws, ws_w=ws, wt=wt,
                           search_abs=search_abs,
                           reflect_bounds=reflect_bounds,
                           use_adj=use_adj, oh0=oh0, ow0=ow0, oh1=oh1, ow1=ow1,
                           remove_self=remove_self, full_ws=full_ws,
                           nbwd=nbwd,rbwd=rbwd, exact=exact)

    def wrap_run(vid0,qstart,nqueries,vid1=None):
        if vid1 is None: vid1 = vid0
        return run_fwd_part(vid0,vid1,qstart,nqueries)

    return wrap_run

# -=-=-=-=-=-=-=-=-=-=-=-=-
#
#          API
#
# -=-=-=-=-=-=-=-=-=-=-=-=-

def init_fwd(nframes,nqueries,ws_h,ws_w,wt,
             k, ps, pt, chnls,
             stride0, stride1, dilation,
             search_abs, reflect_bounds, use_adj,
             oh0, ow0, oh1, ow1, remove_self, full_ws, nbwd,
             rbwd, exact):
    # st = 2*wt+1
    # dists = jnp.zeros((nqueries,st,ws_h,ws_w),dtype=np.float32)
    # inds = jnp.zeros((nqueries,st,ws_h,ws_w,3),dtype=np.int32)
    # tranges = jnp.zeros((nframes,nframes),dtype=np.int32)
    # n_tranges = jnp.ones((nframes),dtype=np.int32)
    # min_tranges = jnp.zeros((nframes),dtype=np.int32)

    # -- init shapes --
    # ishapes = jnp.array([0, nqueries,ws_h,ws_w,wt,
    #                      k, ps, pt, chnls,
    #                      stride0, stride1, dilation,
    #                      search_abs, reflect_bounds, use_adj,
    #                      oh0, ow0, oh1, ow1, remove_self, full_ws, nbwd,
    #                      rbwd, exact, nframes, 0, 0, 0],dtype=jnp.int32)
    run_fwd_part = partial(run_fwd, k=k, ps=ps, pt=pt, chnls=chnls,
                           stride0=stride0, stride1=stride1, dilation=dilation,
                           ws_h=ws_h, ws_w=ws_w, wt=wt,
                           search_abs=search_abs,
                           reflect_bounds=reflect_bounds,
                           use_adj=use_adj, oh0=oh0, ow0=ow0, oh1=oh1, ow1=ow1,
                           remove_self=remove_self, full_ws=full_ws,
                           nbwd=nbwd,rbwd=rbwd, exact=exact)
    # run_fwd_part = partial(run_fwd,dists,inds,tranges,n_tranges,
    #                        min_tranges,ishapes,nqueries,k)
    return run_fwd_part

# @partial(jax.jit,static_argnums=list(range(24)))
def run_fwd(vid0, vid1, qstart, nqueries,
            fflow=None, bflow=None,
            tranges=None, n_tranges=None,
            min_tranges=None, ishapes=None, vshape=None,
            k=5, ps=7, pt=1, chnls=-1,
            stride0=1, stride1=1, dilation=1,
            ws_h=5, ws_w=5, wt=0,
            search_abs=False, reflect_bounds=False, use_adj=False,
            oh0=0, ow0=0, oh1=0, ow1=0, remove_self=False,
            full_ws=False, nbwd=False, rbwd=False, exact=False):

    # -- allocate --
    # st = 2*wt+1
    # dists = jnp.zeros((nqueries,st,ws_h,ws_w),dtype=np.float32)
    # inds = jnp.zeros((nqueries,st,ws_h,ws_w,3),dtype=np.int32)
    # dists = jnp.zeros((nqueries,k),dtype=np.float32)
    # inds = jnp.zeros((nqueries,k,3),dtype=np.int32)

    # -- fill shapes --
    # nframes,nchnls,height,width = vid0.shape
    # ishapes = ishapes.at[-3].set(nchnls)
    # ishapes = ishapes.at[-2].set(height)
    # ishapes = ishapes.at[-1].set(width)
    # ishapes = ishapes.at[0].set(0) # qstart
    # print(ishapes)

    # -- exec cpp primitive --
    # out_dists,out_inds = _primitive_forward.bind(
    #     dists,inds,tranges,n_tranges,min_tranges,ishapes,
    #     nqueries, k, vid0, vid1, fflow, bflow)
    # print(vid0,vid1,qstart,nqueries)
    nframes,color,height,width = vid0.shape
    if tranges is None:
        tranges = jnp.zeros((nframes,nframes),dtype=np.int32)
    if n_tranges is None:
        n_tranges = jnp.ones((nframes),dtype=np.int32)
    if min_tranges is None:
        min_tranges = jnp.zeros((nframes),dtype=np.int32)
    if ishapes is None:
        ishapes = jnp.array([qstart, nqueries,ws_h,ws_w,wt,
                             k, ps, pt, chnls,
                             stride0, stride1, dilation,
                             search_abs, reflect_bounds, use_adj,
                             oh0, ow0, oh1, ow1, remove_self, full_ws, nbwd,
                             rbwd, exact, nframes, color, height, width],
                            dtype=jnp.int32)
    if vshape is None:
        vshape = vid0.shape

    out_dists,out_inds = _primitive_forward.bind(
        vid0, vid1,
        fflow, bflow,
        tranges, n_tranges,
        min_tranges, ishapes,
        # fflow=fflow, bflow=bflow,
        # tranges=tranges, n_tranges=n_tranges,
        # min_tranges=min_tranges, ishapes=ishapes,
        vshape=vshape,
        qstart=qstart, nqueries=nqueries,
        k=k, ps=ps, pt=pt, chnls=chnls,
        stride0=stride0, stride1=stride1, dilation=dilation,
        ws_h=ws_h, ws_w=ws_w, wt=wt,
        search_abs=search_abs,
        reflect_bounds=reflect_bounds, use_adj=use_adj,
        oh0=oh0, ow0=ow0, oh1=oh1, ow1=ow1,
        remove_self=remove_self, full_ws=full_ws,
        nbwd=nbwd,rbwd=rbwd, exact=exact)


    # -- run topk --
    # print("out_dists.shape: ",out_dists.shape)
    # out_dists = jnp.reshape(out_dists,(nqueries,-1))
    # out_inds = jnp.reshape(out_inds,(nqueries,-1,3))
    # order = jnp.argsort(out_dists,1)[:,:k]
    # print("out_dists.shape: ",out_dists.shape)
    # dists_topk = jnp.take_along_axis(out_dists,order,1)
    # inds_topk = []
    # for i in range(3):
    #     inds_topk_i = jnp.take_along_axis(out_inds[:,:,i],order,1)
    #     inds_topk.append(inds_topk_i)
    # inds_topk = jnp.stack(inds_topk,-1)
    dists_topk,inds_topk = out_dists,out_inds

    return dists_topk,inds_topk

def run_bwd(dists,inds,vid0,vid1,fflow,bflow,
            tranges,n_tranges,min_tranges,ishapes,
            qstart=0,ps=1,pt=1,dilation=1,stride0=1,
            oh0=0,ow0=0,oh1=0,ow1=0,use_adj=False,
            reflect_bounds=True,full_ws=True,
            nbwd=1,rbwd=False,exact=True):
    vshape = vid0.shape
    return _primitive_backward.bind(dists,inds,vid0,vid1,fflow,bflow,
                                    tranges,n_tranges,min_tranges,ishapes,
                                    vshape=vshape,
                                    qstart=qstart,ps=ps,pt=pt,dilation=dilation,
                                    stride0=stride0,oh0=oh0,ow0=ow0,oh1=oh1,ow1=ow1,
                                    use_adj=use_adj,reflect_bounds=reflect_bounds,
                                    full_ws=full_ws,nbwd=nbwd,rbwd=rbwd,exact=exact)

# def init(*args,**kwargs): # a "partial" version of run.
#     return run_fwd(*args,**kwargs)

def create_opaque_param(*args):
    nelem = len(args)
    pstr= ("%d" * nelem).format(*args)
    return pstr.encode("ascii")

# -=-=-=-=-=-=-=-=-=-=-=-=-
#
#      Jax Primitives
#
# -=-=-=-=-=-=-=-=-=-=-=-=-

def abstract_forward(vid0, vid1,
                     fflow, bflow,
                     tranges, n_tranges, min_tranges, ishapes,  vshape,
                     # tranges=(1,1), n_tranges=(1,), min_tranges=(1,),
                     # ishapes=(1,), vshape=(1,1,1,1),
                     qstart=0, nqueries=1, k=5, ps=7, pt=1, chnls=-1,
                     stride0=1, stride1=1, dilation=1,
                     ws_h=5, ws_w=5, wt=0,
                     search_abs=False, reflect_bounds=False, use_adj=False,
                     oh0=0, ow0=0, oh1=0, ow1=0, remove_self=False,
                     full_ws=False, nbwd=False, rbwd=False, exact=False):
    st = wt*2 + 1
    f32 = dtypes.canonicalize_dtype(np.float32)
    i32 = dtypes.canonicalize_dtype(np.int32)
    dshape = (nqueries,k)
    ishape = (nqueries,k,3)
    # print("dshape: ",dshape)
    return (ShapedArray(dshape,f32),ShapedArray(ishape,i32))

def forward(xla_builder, vid0, vid1,
            fflow, bflow,
            tranges, n_tranges, min_tranges, ishapes,  vshape,
            # fflow=(1,1,1,1), bflow=(1,1,1,1),
            # tranges=(1,1), n_tranges=(1,),
            # min_tranges=(1,), ishapes=(1,),  vshape=(1,1,1,1),
            qstart=0, nqueries=1,
            k=5, ps=7, pt=1, chnls=-1,
            stride0=1, stride1=1, dilation=1,
            ws_h=5, ws_w=5, wt=0,
            search_abs=False, reflect_bounds=False, use_adj=False,
            oh0=0, ow0=0, oh1=0, ow1=0, remove_self=False, full_ws=False,
            nbwd=False, rbwd=False, exact=False, name=""):

    # -- allocate --
    args = [vid0,vid1,fflow,bflow,tranges,n_tranges,min_tranges,ishapes]
    # print([type(arg) for arg in args])
    # exit(0)

    # -- types --
    f32 = dtypes.canonicalize_dtype(np.float32)
    i32 = dtypes.canonicalize_dtype(np.int32)
    ashape = xla_client.Shape.array_shape

    # -- input --
    input_shapes = []
    for arg in args:
        if isinstance(arg, jnp.ndarray):
            dtype = dtypes.canonicalize_dtype(arg.dtype)
            ndim = arg.ndim
            order = range(ndim-1,-1,-1)
            shape = ashape(dtype,arg.shape,order)
            input_shapes.append(shape)
        else:
            shape = xla_builder.get_shape(arg)
            input_shapes.append(shape)

    # -- output --
    output_shapes = [ashape(f32,(nqueries,k),(1,0))]
    output_shapes += [ashape(i32,(nqueries,k,3),(2,1,0))]
    output_shapes = xla_client.Shape.tuple_shape(output_shapes)

    # -- bytes --
    name = name.encode("ascii")
    opaque = b'..'#create_opaque_param(*oargs)

    # -- exec cpp --
    out = xops.CustomCallWithLayout(
        xla_builder, name, operands=args,
        operand_shapes_with_layout=input_shapes,
        shape_with_layout=output_shapes,
        opaque=opaque
    )

    return out

def abstract_backward(dists,inds,vid0,vid1, fflow, bflow,
                      tranges, n_tranges, min_tranges, ishapes,  vshape,
                      **kwargs):
                      # qstart=qstart,ps=ps,pt=pt,dilation=dilation,
                      # stride0=stride0,oh0=oh0,ow0=ow0,oh1=oh1,ow1=ow1,
                      # use_adj=use_adj,reflect_bounds=reflect_bounds,
                      # full_ws=full_ws,nbwd=nbwd,rbwd=rbwd,exact=exact):
    f32 = dtypes.canonicalize_dtype(np.float32)
    return (ShapedArray(vid0.shape,f32),ShapedArray(vid1.shape,f32))

def backward(xla_builder, dists, inds, vid0, vid1,
             # fflow=(1,1,1,1), bflow=(1,1,1,1),
             # tranges=(1,1), n_tranges=(1,),
             # min_tranges=(1,), ishapes=(1,),  vshape=(1,1,1,1),
             fflow, bflow,
             tranges, n_tranges, min_tranges, ishapes,  vshape,
             qstart=0,ps=1,pt=1,dilation=1,stride0=1,
             oh0=0,ow0=0,oh1=0,ow1=0,use_adj=False,
             reflect_bounds=True,full_ws=True,
             nbwd=1,rbwd=False,exact=True, name=""):

    # -- allocate --
    args = [vid0,vid1,fflow,bflow,tranges,n_tranges,min_tranges,ishapes]

    # -- input --
    input_shapes = [xla_builder.get_shape(arg) for arg in args]
    input_dtypes = tuple(shape.element_type() for shape in input_shapes)
    input_dimensions = tuple(shape.dimensions() for shape in input_shapes)

    # -- output --
    out_shapes = xla_builder.get_shape(vid0),xla_builder.get_shape(vid1)
    output_shapes = xla_client.Shape.tuple_shape(out_shapes)

    # -- view --
    # print(input_shapes)
    name = name.encode("ascii")
    # print(name)
    opaque = b'..'

    # -- exec cpp --
    return xops.CustomCallWithLayout(
        xla_builder,
        name,
        operands=args,
        operand_shapes_with_layout=input_shapes,
        shape_with_layout=output_shapes,
        opaque=opaque
    )

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#      JVP/VJP Use Primitives
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def backward_jvp(arg_values, arg_tangents,
                 # fflow, bflow,
                 # tranges, n_tranges,
                 # min_tranges, ishapes,
                 # fflow=(1,1,1,1), bflow=(1,1,1,1),
                 # tranges=(1,1), n_tranges=(1,),
                 # min_tranges=(1,), ishapes=(1,),
                 qstart=0, nqueries=1, vshape=(1,1,1,1),
                 k=5, ps=7, pt=1, chnls=-1,
                 stride0=1, stride1=1, dilation=1,
                 ws_h=5, ws_w=5, wt=0,
                 search_abs=False, reflect_bounds=False, use_adj=False,
                 oh0=0, ow0=0, oh1=0, ow1=0, remove_self=False,
                 full_ws=False, nbwd=False, rbwd=False, exact=False,
                 fwd_fxn=None, bwd_fxn=None):

    # -- prepare --
    # print("a: ")
    # print([type(arg) for arg in arg_values])
    # print(len(arg_values),arg_values[0].shape)
    # print("b: ")
    # print(len(arg_tangents),arg_tangents[0])#.shape)
    # print(arg_tangents)
    # exit(0)
    vid0,vid1,fflow,bflow,tranges,n_tranges,min_tranges,ishapes = arg_values
    nframes = arg_values[0].shape[0]
    # print("jvp.")


    # -- forward --
    ifwd_fxn = init(fflow, bflow,
                    nframes=nframes, k=k, ps=ps, pt=pt, chnls=chnls,
                    stride0=stride0, stride1=stride1, dilation=dilation,
                    ws=ws_h, wt=wt,
                    search_abs=search_abs, reflect_bounds=reflect_bounds,
                    use_adj=use_adj, oh0=oh0, ow0=ow0,
                    oh1=oh1, ow1=ow1, remove_self=remove_self,
                    full_ws=full_ws, nbwd=nbwd, rbwd=rbwd, exact=exact)
    dists,inds = ifwd_fxn(vid0,qstart,nqueries,vid1)

    # -- tangent (backward at point) --
    # print("pre tan.")
    vid0_grad,vid1_grad = run_bwd(dists,inds,vid0,vid1,fflow,bflow,
                                  tranges,n_tranges,min_tranges,ishapes,
                                  qstart=qstart,ps=ps,pt=pt,dilation=dilation,
                                  stride0=stride0,oh0=oh0,ow0=ow0,oh1=oh1,ow1=ow1,
                                  use_adj=use_adj,reflect_bounds=reflect_bounds,
                                  full_ws=full_ws,nbwd=nbwd,rbwd=rbwd,exact=exact)
    # print("post tan.")

    return ((dists,inds), (vid0_grad,vid1_grad))

def backward_vjp(*args,qstart=0, nqueries=1, vshape=(1,1,1,1),
                 k=5, ps=7, pt=1, chnls=-1,
                 stride0=1, stride1=1, dilation=1,
                 ws_h=5, ws_w=5, wt=0,
                 search_abs=False, reflect_bounds=False, use_adj=False,
                 oh0=0, ow0=0, oh1=0, ow1=0, remove_self=False,
                 full_ws=False, nbwd=False, rbwd=False, exact=False,
                 fwd_fxn=None, bwd_fxn=None):
# **kwargs,name_fwd="",name_bwd=""): # "backward"

    # -- init --
    print("vjp.")
    print("a: ")
    print(len(args))
    exit(0)
    # vid0,vid1,fflow,bflow,tranges,n_tranges,min_tranges,ishapes = arg_values
    # nframes = arg_values[0].shape[0]

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
    jvp = partial(backward_jvp,fwd_fxn=run_fwd,bwd_fxn=run_bwd)
    vjp = partial(backward_vjp,fwd_fxn=run_fwd,bwd_fxn=run_bwd)

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
    ad.primitive_transposes[prim_fwd] = vjp

_register() # call for init
