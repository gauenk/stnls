
import sys, os
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)


from functools import partial
from jax.lib import xla_client

import numba
import numpy as np
from numba import types as nb_types
from ._method_cache import get_custom_call_name
from . import xla_utils
from cffi import FFI



xla_call_sig = nb_types.void(
    nb_types.voidptr,  # cudaStream_t* stream
    nb_types.CPointer(nb_types.voidptr),  # void** buffers
    nb_types.voidptr,  # const char* opaque
    nb_types.uint64,  # size_t opaque_len
)


def compile_gpu_signature(
    gpu_fn, *, input_shapes, input_dtypes, output_shapes, output_dtypes
):
    """
    Compiles gpu_fn to C and register it with XLA for the given signature.
    """

    from ._cuda import (
        cuMemcpy,
        cuMemcpyAsync,
        cuStreamSynchronize,
        memcpyHostToHost,
        memcpyHostToDevice,
        memcpyDeviceToHost,
        memcpyDeviceToDevice,
    )
    print("cuMemcpy: ",cuMemcpy)

    n_in = len(input_shapes)
    n_out = len(output_shapes)
    input_byte_size = tuple(
        np.prod(shape) * dtype.itemsize
        for (shape, dtype) in zip(input_shapes, input_dtypes)
    )
    output_byte_size = tuple(
        np.prod(shape) * dtype.itemsize
        for (shape, dtype) in zip(output_shapes, output_dtypes)
    )

    # -- ffi --
    # ffi = FFI()
    # ffi.cdef("void _Z11simple_testPvS_();")
    # fn = "/home/gauenk/Documents/packages/stnls/lib/stnls_cuda.cpython-38-x86_64-linux-gnu.so"
    # print(ffi.RTLD_GLOBAL,ffi.RTLD_LOCAL)
    # lib = ffi.dlopen(fn,flags=ffi.RTLD_GLOBAL | ffi.RTLD_LOCAL | ffi.RTLD_DEEPBIND)
    # print(lib)
    # print(dir(lib))
    # stest = lib._Z11simple_testPvS_
    # print(stest)

    # print(gpu_fn,stest)
    # print("gpu_fn: ",gpu_fn)

    # print(dir(gpu_fn))
    # @numba.njit()
    # def test_this(args):
    #     # gpu_fn(args)
        # return None
    # print("test_this: ",test_this)
    # @numba.jit()
    # def test_this(*args):
    #     gpu_fn(*args)

    @numba.cfunc(xla_call_sig)
    def xla_custom_call_target(stream, io_gpu_ptrs, opaque, opaque_len):
        gpu_fn(io_gpu_ptrs[0],io_gpu_ptrs[0])

    target_name = xla_custom_call_target.native_name.encode("ascii")

    # Extract the pointer to the CFFI function and create a pycapsule
    # around it
    capsule = xla_utils.create_xla_target_capsule(xla_custom_call_target.address)

    print("target_name: ",target_name)
    xla_client.register_custom_call_target(target_name, capsule, "gpu")

    return target_name


def xla_encode(gpu_fn, abstract_eval_fn, xla_builder, *args):
    # if not cuda.numba_cffi_loaded:
    #     raise RuntimeError("Numba cffi could not be loaded.")

    input_shapes = [xla_builder.get_shape(arg) for arg in args]
    input_dtypes = tuple(shape.element_type() for shape in input_shapes)
    input_dimensions = tuple(shape.dimensions() for shape in input_shapes)

    # TODO(josipd): Check that the input layout is the numpy default.
    output_abstract_arrays = abstract_eval_fn(
        *[xla_utils.xla_shape_to_abstract(shape) for shape in input_shapes]
    )
    output_shapes = tuple(array.shape for array in output_abstract_arrays)
    output_dtypes = tuple(array.dtype for array in output_abstract_arrays)

    output_layouts = map(lambda shape: range(len(shape) - 1, -1, -1), output_shapes)

    xla_output_shapes = [
        xla_client.Shape.array_shape(*arg)
        for arg in zip(output_dtypes, output_shapes, output_layouts)
    ]
    xla_output_shape = xla_client.Shape.tuple_shape(xla_output_shapes)

    target_name = get_custom_call_name(
        "gpu",
        gpu_fn,
        input_shapes=input_dimensions,
        input_dtypes=input_dtypes,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        compile_fun=compile_gpu_signature,
    )

    return xla_client.ops.CustomCallWithLayout(
        xla_builder,
        target_name,
        operands=args,
        shape_with_layout=xla_output_shape,
        operand_shapes_with_layout=input_shapes,
    )
