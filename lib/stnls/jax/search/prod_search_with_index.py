3
# -- linalg helpers --
import torch as th # for base cpp
import numpy as np # help
from functools import partial

# -- base cpp --
import stnls_cuda

# -- linking --
from ..link import xla_utils,primitive_utils
from ..link.xla_utils import trace
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

    # @partial(jax.jit,static_argnums=(1,2))
    def wrap_run(vid0,qstart,nqueries,vid1=None):
        if vid1 is None: vid1 = vid0
        part2 = partial(run_fwd_part,qstart=qstart,nqueries=nqueries)
        # print("hi.")
        return part2(vid0,vid1)


    # def wrap_run(qstart,nqueries):


    #     # -- custom partial --
    #     part2 = partial(run_fwd,qstart=qstart,nqueries=nqueries,
    #                     fflow=fflow,bflow=bflow,
    #                     k=k, ps=ps, pt=pt, chnls=chnls,
    #                     stride0=stride0, stride1=stride1, dilation=dilation,
    #                     ws_h=ws, ws_w=ws, wt=wt,
    #                     search_abs=search_abs,
    #                     reflect_bounds=reflect_bounds,
    #                     use_adj=use_adj, oh0=oh0, ow0=ow0, oh1=oh1, ow1=ow1,
    #                     remove_self=remove_self, full_ws=full_ws,
    #                     nbwd=nbwd,rbwd=rbwd, exact=exact)
    #     part2 = custom_vjp(part2)

    #     # -- create partial --
    #     jvp_fwd = partial(forward_jvp,qstart=qstart,nqueries=nqueries,
    #                       fflow=fflow, bflow=bflow,
    #                       nframes=nframes, k=k, ps=ps, pt=pt, chnls=chnls,
    #                       stride0=stride0, stride1=stride1, dilation=dilation,
    #                       ws=ws, wt=wt, search_abs=search_abs,
    #                       reflect_bounds=reflect_bounds,
    #                       use_adj=use_adj, oh0=oh0, ow0=ow0,
    #                       oh1=oh1, ow1=ow1, remove_self=remove_self,
    #                       full_ws=full_ws, nbwd=nbwd, rbwd=rbwd, exact=exact)
    #     vjp_fwd = partial(forward_vjp,
    #                       qstart=qstart,nqueries=nqueries,
    #                       k=k, ps=ps, pt=pt, chnls=chnls,
    #                       stride0=stride0, stride1=stride1, dilation=dilation,
    #                       ws=ws, wt=wt, search_abs=search_abs,
    #                       reflect_bounds=reflect_bounds,
    #                       use_adj=use_adj, oh0=oh0, ow0=ow0,
    #                       oh1=oh1, ow1=ow1, remove_self=remove_self,
    #                       full_ws=full_ws, nbwd=nbwd, rbwd=rbwd, exact=exact)
    #     part2.defvjp(jvp_fwd,vjp_fwd)

    #     return part2

    # -- define jvp/vjp --
    # jvp_fwd = forward_jvp
    # vjp_fwd = forward_vjp
    # wrap_run.defvjp(jvp_fwd,vjp_fwd)

    return wrap_run

# -=-=-=-=-=-=-=-=-=-=-=-=-
#
#          API
#
# -=-=-=-=-=-=-=-=-=-=-=-=-

from jax import custom_vjp

# @trace("init_fwd")
# @custom_vjp
def init_fwd(nframes=0,nqueries=0,ws_h=0,ws_w=0,wt=0,
             k=0, ps=0, pt=0, chnls=-1,
             stride0=1, stride1=1, dilation=1,
             search_abs=False, reflect_bounds=False, use_adj=False,
             oh0=0, ow0=0, oh1=0, ow1=0, remove_self=False,
             full_ws=False, nbwd=1, rbwd=False, exact=False):
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
def run_fwd(vid0, vid1, qstart=0, nqueries=1,
            fflow=None, bflow=None,
            tranges=None, n_tranges=None,
            min_tranges=None, ishapes=None, vshape=None,
            k=5, ps=7, pt=1, chnls=-1,
            stride0=1, stride1=1, dilation=1,
            ws_h=5, ws_w=5, wt=0,
            search_abs=False, reflect_bounds=False, use_adj=False,
            oh0=0, ow0=0, oh1=0, ow1=0, remove_self=False,
            full_ws=False, nbwd=1, rbwd=False, exact=False):

    if fflow is None:
        # print("b.")
        exit(0)

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
        # print(qstart, nqueries,ws_h,ws_w,wt,
        #       k, ps, pt, chnls,
        #       stride0, stride1, dilation,
        #       search_abs, reflect_bounds, use_adj,
        #       oh0, ow0, oh1, ow1, remove_self, full_ws, nbwd,
        #       rbwd, exact, nframes, color, height, width)
        ishapes = jnp.array([qstart, nqueries,ws_h,ws_w,wt,
                             k, ps, pt, chnls,
                             stride0, stride1, dilation,
                             search_abs, reflect_bounds, use_adj,
                             oh0, ow0, oh1, ow1, remove_self, full_ws, nbwd,
                             rbwd, exact, nframes, color, height, width],
                            dtype=jnp.int32)
    # print("\n"*5)
    # print("ishapes: ",type(ishapes))
    # print("\n"*5)
    if vshape is None:
        vshape = vid0.shape
    # print("[run_fwd] type(ishapes): ",type(ishapes))


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

def init_bwd(ishapes,ps,pt,dilation=1,
             stride0=1,oh0=0,ow0=0,oh1=0,ow1=0,
             use_adj=False,reflect_bounds=True,
             full_ws=False,nbwd=1,rbwd=False,exact=False):
    run_bwd_part = partial(run_bwd,ishapes=ishapes,
                           ps=ps,pt=pt,dilation=dilation,
                           stride0=stride0,oh0=oh0,ow0=ow0,oh1=oh1,ow1=ow1,
                           use_adj=use_adj,reflect_bounds=reflect_bounds,
                           full_ws=full_ws,nbwd=nbwd,rbwd=rbwd,exact=exact)
    return run_bwd_part


# @trace("run_bwd")
def run_bwd(dists,inds,vid0,vid1,
            qstart=0,
            ishapes=(1,),
            ps=1,pt=1,dilation=1,stride0=1,
            oh0=0,ow0=0,oh1=0,ow1=0,use_adj=False,
            reflect_bounds=True,full_ws=True,
            nbwd=1,rbwd=False,exact=True):
    # print("yo.")
    # print(dists,inds,vid0,vid1)
    nqueries,k = dists.shape
    nframes,color,height,width = vid0.shape
    if ishapes is None:
        # print(qstart, nqueries,ws_h,ws_w,wt,
        #       k, ps, pt, chnls,
        #       stride0, stride1, dilation,
        #       search_abs, reflect_bounds, use_adj,
        #       oh0, ow0, oh1, ow1, remove_self, full_ws, nbwd,
        #       rbwd, exact, nframes, color, height, width)
        ishapes = jnp.array([qstart, nqueries,0,0,0,
                             k, ps, pt, 0,
                             stride0, 0, dilation,
                             0, reflect_bounds, use_adj,
                             oh0, ow0, oh1, ow1, False, full_ws, nbwd,
                             rbwd, exact, nframes, color, height, width],
                            dtype=jnp.int32)
    # if vshape is None:
    #     vshape = vid0.shape

    # vshape = vid0.shape
    # return backward(dists,inds,vid0,vid1,
    #                 qstart=qstart,ishapes=ishapes,
    #                 ps=ps,pt=pt,dilation=dilation,
    #                 stride0=stride0,oh0=oh0,ow0=ow0,oh1=oh1,ow1=ow1,
    #                 use_adj=use_adj,reflect_bounds=reflect_bounds,
    #                 full_ws=full_ws,nbwd=nbwd,rbwd=rbwd,exact=exact)

    vid_grads = _primitive_backward.bind(dists,inds,vid0,vid1,
                                    ishapes, qstart=qstart,
                                    # vshape=vshape,
                                    ps=ps,pt=pt,dilation=dilation,
                                    stride0=stride0,oh0=oh0,ow0=ow0,oh1=oh1,ow1=ow1,
                                    use_adj=use_adj,reflect_bounds=reflect_bounds,
                                    full_ws=full_ws,nbwd=nbwd,rbwd=rbwd,exact=exact)
    return vid_grads

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

def abstract_forward(vid0, vid1, fflow, bflow,
                     tranges, n_tranges, min_tranges, ishapes,  vshape,
                     # tranges=(1,1), n_tranges=(1,), min_tranges=(1,),
                     # ishapes=(1,), vshape=(1,1,1,1),
                     qstart=0, nqueries=1, k=5, ps=7, pt=1, chnls=-1,
                     stride0=1, stride1=1, dilation=1,
                     ws_h=5, ws_w=5, wt=0,
                     search_abs=False, reflect_bounds=False, use_adj=False,
                     oh0=0, ow0=0, oh1=0, ow1=0, remove_self=False,
                     full_ws=False, nbwd=1, rbwd=False, exact=False):
    st = wt*2 + 1
    f32 = dtypes.canonicalize_dtype(np.float32)
    i32 = dtypes.canonicalize_dtype(np.int32)
    dshape = (nqueries,k)
    ishape = (nqueries,k,3)
    # print("dshape: ",dshape)
    return (ShapedArray(dshape,f32),ShapedArray(ishape,i32))

# @trace("forward")
# @custom_vjp
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
            nbwd=1, rbwd=False, exact=False, name=""):

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

def abstract_backward(dists,inds,vid0,vid1,ishapes,
                      # fflow, bflow,
                      # tranges, n_tranges, min_tranges,
                      # ishapes,
                      # vshape,
                      **kwargs):
                      # qstart=qstart,ps=ps,pt=pt,dilation=dilation,
                      # stride0=stride0,oh0=oh0,ow0=ow0,oh1=oh1,ow1=ow1,
                      # use_adj=use_adj,reflect_bounds=reflect_bounds,
                      # full_ws=full_ws,nbwd=nbwd,rbwd=rbwd,exact=exact):
    f32 = dtypes.canonicalize_dtype(np.float32)
    return (ShapedArray(vid0.shape,f32),ShapedArray(vid1.shape,f32))

# @trace("backward")
def backward(xla_builder,
             dists, inds, vid0, vid1, ishapes,
             qstart=0,
             # fflow=(1,1,1,1), bflow=(1,1,1,1),
             # tranges=(1,1), n_tranges=(1,),
             # min_tranges=(1,),
             # ishapes=(1,),
             # vshape=(1,1,1,1),
             # fflow, bflow,
             # tranges, n_tranges, min_tranges, ishapes,  vshape,
             ps=1,pt=1,dilation=1,stride0=1,
             oh0=0,ow0=0,oh1=0,ow1=0,use_adj=False,
             reflect_bounds=True,full_ws=True,
             nbwd=1,rbwd=False,exact=True, name=""):

    # -- allocate --
    # args = [dists,inds,vid0,vid1,fflow,bflow,tranges,n_tranges,min_tranges,ishapes]
    # ishapes = jnp.array([0,0],dtype=jnp.int32)
    # print("type(ishapes): ",type(ishapes))
    args = [dists,inds,vid0,vid1,ishapes]
    # print("dists: ",dists)
    # print("inds: ",inds)
    # print("vid0: ",vid0)
    # print("vid1: ",vid1)
    # print("ishapes: ",ishapes)
    # print(dists)

    # -- input --
    f32 = dtypes.canonicalize_dtype(np.float32)
    i32 = dtypes.canonicalize_dtype(np.int32)
    ashape = xla_client.Shape.array_shape
    input_shapes = []
    for arg in args:
        if isinstance(arg, jnp.ndarray):
            # print(args)
            dtype = dtypes.canonicalize_dtype(arg.dtype)
            ndim = arg.ndim
            order = range(ndim-1,-1,-1)
            shape = ashape(dtype,arg.shape,order)
            input_shapes.append(shape)
        # elif hasattr(arg,"shape"):
        #     shape = arg.shape
        #     input_shapes.append(shape)
        else:
            shape = xla_builder.get_shape(arg)
            # shape = arg.shape
            input_shapes.append(shape)
    # input_shapes = [arg.shape for arg in args]
    # input_shapes = [xla_builder.get_shape(arg) for arg in args]
    # input_dtypes = tuple(shape.element_type() for shape in input_shapes)
    # input_dimensions = tuple(shape.dimensions() for shape in input_shapes)
    # print(input_shapes)

    # -- output --
    # out_shapes = vid0.shape,vid1.shape
    out_shapes = xla_builder.get_shape(vid0),xla_builder.get_shape(vid1)
    output_shapes = xla_client.Shape.tuple_shape(out_shapes)
    # output_shapes = out_shapes
    # print(output_shapes)
    # exit(0)

    # -- view --
    # print(input_shapes)
    name = name.encode("ascii")
    # print(name)
    opaque = b'..'

    # -- exec cpp --
    cpp_out = xops.CustomCallWithLayout(
        xla_builder,
        name,
        operands=args,
        operand_shapes_with_layout=input_shapes,
        shape_with_layout=output_shapes,
        opaque=opaque
    )

    return cpp_out

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#      JVP/VJP Use Primitives
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# @trace("forward_jvp")
def forward_jvp(# arg_values, arg_tangents,
                 # fflow, bflow,
                 # tranges, n_tranges,
                 # min_tranges,
                 # ishapes,
                 # fflow=(1,1,1,1), bflow=(1,1,1,1),
                 # tranges=(1,1), n_tranges=(1,),
                 # min_tranges=(1,),
        vid0, vid1,
        qstart=0, nqueries=1,
        fflow=None, bflow=None,
        tranges=None, n_tranges=None, min_tranges=None,# ishapes,  vshape,
        ishapes=None, vshape=None,
                 # ishapes=(1,),
                 # qstart=0, nqueries=1,
                 # vshape=(1,1,1,1),
                 k=5, ps=7, pt=1, chnls=-1,
                 stride0=1, stride1=1, dilation=1,
                 ws_h=5, ws_w=5, wt=0,
                 search_abs=False, reflect_bounds=False, use_adj=False,
                 oh0=0, ow0=0, oh1=0, ow1=0, remove_self=False,
                 full_ws=False, nbwd=1, rbwd=False, exact=False):
                 # fwd_fxn=None, bwd_fxn=None):

    # -- prepare --
    # print("a: ")
    # print([type(arg) for arg in arg_values])
    # print(len(arg_values),arg_values[0].shape)
    # print("b: ")
    # print(len(arg_tangents),arg_tangents[0])#.shape)
    # print(arg_tangents)
    # exit(0)
    # vid0,vid1,fflow,bflow,tranges,n_tranges,min_tranges,ishapes = arg_values
    # vid0,vid1,fflow,bflow,tranges,n_tranges,min_tranges,ishapes = arg_tangents
    nframes = vid0.shape[0]
    # print("jvp.")
    # print(        vid0, vid1,
    #               fflow, bflow,
    #               tranges, n_tranges, min_tranges,# ishapes,  vshape,
    # )

    # print("--"*30)
    # print(fflow, bflow,
    #       tranges, n_tranges, min_tranges,
    #       ishapes, qstart, nqueries,
    #       vshape,k, ps, pt, chnls,
    #       stride0, stride1, dilation,
    #       ws_h, ws_w, wt, search_abs, reflect_bounds, use_adj,
    #       oh0, ow0, oh1, ow1, remove_self,
    #       full_ws, nbwd, rbwd, exact)

    # print(ishapes)
    # print(qstart,nqueries)
    # qstart = 0
    # nqueries = 1024
    vshape = vid0.shape
    # print(qstart,nqueries)
    # if fflow is None:
    #     exit(0)
    # if ps is None:
    #     exit(0)
    # exit(0)

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

    # print(dists[:3,:3])
    # exit(0)
    # dists_2,inds_2 = ifwd_fxn(vid0,qstart,nqueries,vid1)
    # vid0,vid1,_,_,_,_,_,_ = arg_tangents
    # vid0,vid1,_,_,_,_,_,ishapes = arg_tangents
    # vid0,vid1,_,_,_,_,_,_ = arg_values

    # -- tangent (backward at point) --
    # print("pre tan.")
    # print("tans: ",len(arg_tangents))
    # print("\n"*20)
    # print(arg_tangents[0])
    # _ishapes = jnp.zeros((28,))
    # print(ishapes,dists,inds)
    # print("here: ",vid0,vid1)
    # exit(0)

    # -- this one --
    # ibwd_fxn = init_bwd(ishapes,ps=ps,pt=pt,dilation=dilation,
    #                     stride0=stride0,oh0=oh0,ow0=ow0,oh1=oh1,ow1=ow1,
    #                     use_adj=use_adj,reflect_bounds=reflect_bounds,
    #                     full_ws=full_ws,nbwd=nbwd,rbwd=rbwd,exact=exact)
    # vid0_grad,vid1_grad = ibwd_fxn(dists_2,inds_2,vid0,vid1,qstart)
    # vid_grads = (vid0_grad,vid1_grad)
    # nframes,color,height,width = vid0.shape

    # -- STOP HERE --
    # ishapes = jnp.array([qstart, nqueries,0,0,0,
    #                      0, ps, pt, 0,
    #                      stride0, 0, dilation,
    #                      0, reflect_bounds, use_adj,
    #                      oh0, ow0, oh1, ow1, False, full_ws, nbwd,
    #                      rbwd, exact, nframes, color, height, width],
    #                     dtype=jnp.int32)
    # ishapes = jnp.array([0,0],dtype=jnp.int32)

    # vid_grads = _primitive_backward.bind(dists_2,inds_2,vid0,vid1,
    #                                      ishapes,qstart=qstart,#ishapes=ishapes,
    #                                 # vshape=vshape,
    #                                 ps=ps,pt=pt,dilation=dilation,
    #                                 stride0=stride0,oh0=oh0,ow0=ow0,oh1=oh1,ow1=ow1,
    #                                 use_adj=use_adj,reflect_bounds=reflect_bounds,
    #                                 full_ws=full_ws,nbwd=nbwd,rbwd=rbwd,exact=exact)
    # print("post: ",vid_grads)
    # vid0_grad = jnp.zeros_like(vid0)
    # vid1_grad = jnp.zeros_like(vid0)
    # vid_grads = (vid0_grad,vid1_grad)

    # print("post tan.")
    # print(vid0_grad,vid1_grad)
    # exit(0)

    return (dists,inds), (vid0,vid1,inds)
    # return (dists,inds), vid_grads
    # return ((dists,inds), (vid0_grad,vid1_grad))

# # @trace("forward_vjp")
# def forward_vjp(vids,y_bar,
#         # qstart=0, nqueries=1,
#         # fflow=None, bflow=None,
#         # tranges=None, n_tranges=None, min_tranges=None,# ishapes,  vshape,
#         # ishapes=None, vshape=None,
#                  # ishapes=(1,),
#                  qstart=0, nqueries=1,
#                 # vshape=(1,1,1,1),
#                  k=5, ps=7, pt=1, chnls=-1,
#                 stride0=1, stride1=1, dilation=1,
#                 ws_h=5, ws_w=5, wt=0,
#                 search_abs=False, reflect_bounds=False, use_adj=False,
#                 oh0=0, ow0=0, oh1=0, ow1=0, remove_self=False,
#                 full_ws=False, nbwd=False, rbwd=False, exact=False):
# # **kwargs,name_fwd="",name_bwd=""): # "backward"

# def forward_vjp(vids,y_bar,
#         qstart=0, nqueries=1,
#         fflow=None, bflow=None,
#         tranges=None, n_tranges=None, min_tranges=None,# ishapes,  vshape,
#         ishapes=None, vshape=None,
#                  # ishapes=(1,),
#                  # qstart=0, nqueries=1,
#                  # vshape=(1,1,1,1),
#                  k=5, ps=7, pt=1, chnls=-1,
#                  stride0=1, stride1=1, dilation=1,
#                  ws_h=5, ws_w=5, wt=0,
#                  search_abs=False, reflect_bounds=False, use_adj=False,
#                  oh0=0, ow0=0, oh1=0, ow1=0, remove_self=False,
#                  full_ws=False, nbwd=False, rbwd=False, exact=False):

# qstart=0, nqueries=1,
#             fflow=None, bflow=None,
#             tranges=None, n_tranges=None,
#             min_tranges=None, ishapes=None, vshape=None,
#             k=5, ps=7, pt=1, chnls=-1,
#             stride0=1, stride1=1, dilation=1,
#             ws_h=5, ws_w=5, wt=0,
#             search_abs=False, reflect_bounds=False, use_adj=False,
#             oh0=0, ow0=0, oh1=0, ow1=0, remove_self=False,
#             full_ws=False, nbwd=1, rbwd=False, exact=False
def forward_vjp(qstart, nqueries,
                fflow, bflow,
                tranges, n_tranges, min_tranges,
                ishapes, vshape,
                k, ps, pt, chnls,
                stride0, stride1, dilation,
                ws_h, ws_w, wt,
                search_abs, reflect_bounds, use_adj,
                oh0, ow0, oh1, ow1, remove_self,
                full_ws, nbwd, rbwd, exact, vids, cotan):

            # fflow=None, bflow=None,
            # tranges=None, n_tranges=None,
            # min_tranges=None, ishapes=None, vshape=None,
            # k=5, ps=7, pt=1, chnls=-1,
            # stride0=1, stride1=1, dilation=1,
            # ws_h=5, ws_w=5, wt=0,
            # search_abs=False, reflect_bounds=False, use_adj=False,
            # oh0=0, ow0=0, oh1=0, ow1=0, remove_self=False,
            # full_ws=False, nbwd=1, rbwd=False, exact=False):

    # -- init --
    # print("this.")
    # print(vids)
    # print(y_bar)
    # print(qstart.shape,nqueries.shape)
    # print(qstart,nqueries)
    # print(fflow.shape,bflow.shape)
    # print(tranges,n_tranges,min_tranges,ishapes,vshape)
    # print(k, ps, pt, chnls,
    #       stride0, stride1, dilation,
    #       ws_h, ws_w, wt)
    # print(search_abs, reflect_bounds, use_adj,
    #       oh0, ow0, oh1, ow1, remove_self,
    #       full_ws, nbwd, rbwd)
    # print(exact)
    # print(vids)
    # print(cotan)
    vid0,vid1,inds = vids
    dists,_ = cotan
    # print(dists[:3,:3])

    # exit(0)
    # vid0,vid1 = vids
    # dists,inds = y_bar
    # print("vjp.")
    # print("a: ")
    # print(len(args))
    # print(len(args[0]),len(args[1]))
    # print(args[0])
    # print(args[1])
    # print(vid0.shape,vid1.shape)
    # print(dists,inds)
    # exit(0)

    # print("vjp.")
    # print("\n"*30)
    # print(dists)
    # print(vid0)

    # qstart=0
    # ishapes = None
    # print(vid0[0,0,:3,:3])
    # print(vid1[0,0,:3,:3])
    # exit(0)
    ibwd_fxn = init_bwd(ishapes,ps=ps,pt=pt,dilation=dilation,
                        stride0=stride0,oh0=oh0,ow0=ow0,oh1=oh1,ow1=ow1,
                        use_adj=use_adj,reflect_bounds=reflect_bounds,
                        full_ws=full_ws,nbwd=nbwd,rbwd=rbwd,exact=exact)
    vid0_grad,vid1_grad = ibwd_fxn(dists,inds,vid0,vid1,qstart)
    # vid_grads = (vid0_grad,vid1_grad)
    # nframes,color,height,width = vid0.shape


    # exit(0)
    # exit(0)
    # vid0,vid1,fflow,bflow,tranges,n_tranges,min_tranges,ishapes = arg_values
    # nframes = arg_values[0].shape[0]

    # -- primal --
    # fwd_out = run_fwd(*arg_values)

    # # -- tangent --
    # def make_zero(tan):
    #     return lax.zeros_like_array(x) if type(tan) is ad.Zero else tan
    # bwd_out = run_bwd(*arg_values) # output_tangent
    # return (fwd_out, bwd_out)

    return (vid0_grad,vid1_grad)# + (None,)*31

def backward_jvp(*args,**kwargs):
    # print(len(args))
    # print(args[0].shape)
    print("jvp.")
    exit(0)
    return args[0]

def backward_transpose(ct_dists_inds,dists,inds,vid0,vid1,
                       ishapes=(1,),qstart=0,ps=1,pt=1,dilation=1,
                       stride0=1,oh0=0,ow0=0,oh1=0,ow1=0,
                       use_adj=False,reflect_bounds=True,
                       full_ws=False,nbwd=1,rbwd=False,exact=False,
                       fwd_fxn=None,bwd_fxn=None):
    # print(ct_dists_inds)
    dists_ct,inds_ct = ct_dists_inds
    # print(dists.shape)
    # print(inds.shape)
    # print(dists,inds)
    # print(vid0,vid1)
    # print("btrans.")
    # ibwd_fxn = init_bwd(ishapes,ps=ps,pt=pt,dilation=dilation,
    #                     stride0=stride0,oh0=oh0,ow0=ow0,oh1=oh1,ow1=ow1,
    #                     use_adj=use_adj,reflect_bounds=reflect_bounds,
    #                     full_ws=full_ws,nbwd=nbwd,rbwd=rbwd,exact=exact)
    # vid0_grad,vid1_grad = ibwd_fxn(dists,inds,vid0,vid1,qstart)

    # # print(args[0].shape)
    # exit(0)
    # return args[0]
    # return dists,inds,vid0,vid1
    return None,None,vid0,vid1

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Jax Registration
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def _register():

    # -- assign primitive for api --
    global _primitive_forward
    global _primitive_backward
    global init_fwd

    # -- unpack c++ --
    name = "search_prod_with_index_jax"
    pair = stnls_cuda.search_prod_with_index_jax()
    fwd_cpp,bwd_cpp = pair['forward'],pair['backward']

    # -- register c++ --
    name_fwd,name_bwd = primitive_utils.xla_register(name, fwd_cpp, bwd_cpp)

    # -- wrap --
    fwd = partial(forward,name=name_fwd)
    # fwd = custom_vjp(fwd)
    bwd = partial(backward,name=name_bwd)
    jvp_fwd = forward_jvp#partial(forward_jvp,fwd_fxn=run_fwd,bwd_fxn=run_bwd)
    vjp_fwd = forward_vjp#partial(forward_vjp,fwd_fxn=run_fwd,bwd_fxn=run_bwd)
    jvp_bwd = partial(backward_jvp,fwd_fxn=run_fwd,bwd_fxn=run_bwd)
    transpose_bwd = partial(backward_transpose,fwd_fxn=run_fwd,bwd_fxn=run_bwd)

    # -- define primitive --
    name = "search_prod_with_index_jax"
    prim_fwd = primitive_utils.cfunc_to_jax(name,fwd,abstract_forward,batching_fn=None)
    name = "search_prod_with_index_jax_backward"
    prim_bwd=primitive_utils.cfunc_to_jax(name,bwd,abstract_backward,batching_fn=None)

    # -- assign primitive for api --
    _primitive_forward = prim_fwd
    _primitive_backward = prim_bwd

    # -- assign jvp/vjp for forward --
    # ad.primitive_jvps[prim_fwd] = jvp_fwd
    # ad.primitive_transposes[prim_fwd] = vjp_fwd
    ad.primitive_jvps[prim_bwd] = jvp_bwd
    ad.primitive_transposes[prim_bwd] = transpose_bwd # vjp


    # -- assign jvp/vjp --
    # run_fwd.defvjp(jvp_fwd,vjp_fwd)
    # init_fwd.defvjp(jvp_fwd,vjp_fwd)
    # init_fwd.defvjp(vjp,vjp)
    # ad.defvjp_all(_primitive_forward,vjp_fwd)
    # ad.defvjp(_primitive_forward,vjp_fwd)
    # init_fwd.defvjp(vjp,vjp)
    # f_fwd, f_bwd)

_register() # call for init

# _primitive_forward = custom_vjp(_primitive_forward)

run_fwd = custom_vjp(run_fwd,nondiff_argnums=list(range(2,33)))
jvp_fwd = forward_jvp
vjp_fwd = forward_vjp
# _primitive_forward.defvjp(jvp_fwd,vjp_fwd)
run_fwd.defvjp(jvp_fwd,vjp_fwd)
# init_fwd.defvjp(jvp_fwd,vjp_fwd)

# jvp_fwd = forward_jvp
# vjp_fwd = forward_vjp
# forward.defvjp(jvp_fwd,vjp_fwd)
# ad.defvjp(_primitive_forward.defvjp,vjp_fwd)
