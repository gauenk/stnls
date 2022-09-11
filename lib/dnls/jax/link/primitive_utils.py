"""

Initial Code from:
https://github.com/PhilipVinc/numba4jax/

"""

import torch as th
import collections
import ctypes
from functools import partial  # pylint:disable=g-importing-member
from textwrap import dedent

import jax
import jax.numpy as jnp
from jax.interpreters import batching
from jax.interpreters import xla
from jax.interpreters import ad
from jax.lib import xla_client


import numba
from numba import types as nb_types
import numpy as np

from . import xla_utils
# from . import gpu



def eval_rule(call_fn, abstract_eval_fn, *args, **kwargs):
    """
    Python Evaluation rule for a numba4jax function respecting the
    XLA CustomCall interface.
    Evaluates `outs = abstract_eval_fn(*args)` to compute the output shape
    and preallocate them, then executes `call_fn(*outs, *args)` which is
    the Numba kernel.
    Args:
        call_fn: a (numba.jit) function respecting the calling convention of
            XLA CustomCall, taking first the outputs by reference then the
            inputs.
        abstract_eval_fn: The abstract evaluation function respecting jax
            interface
        args: The arguments to the `call_fn`
        kwargs: Optional keyword arguments for the numba function.
    """
    import ctypes

    # compute the output shapes
    output_shapes = abstract_eval_fn(*args)
    # Preallocate the outputs
    outputs = tuple(np.empty(shape.shape, dtype=shape.dtype) for shape in output_shapes)
    # convert inputs to a tuple
    inputs = tuple(np.asarray(arg) for arg in args)
    args = outputs+inputs
    args = [th.from_numpy(arg).to("cuda:0") for arg in args]
    # call the kernel
    print(outputs + inputs,inputs,outputs,call_fn)
    # print(args[0])
    call_fn(args[0],args[1])
    # call_fn(outputs + inputs, **kwargs)
    # Return the outputs
    return tuple(outputs)


def naive_batching_rule(call_fn, args, batch_axes):
    """
    Returns the batching rule for a numba4jax kernel, which simply
    maps the call among the batched axes.
    """
    # TODO(josipd): Check that the axes are all zeros. Add support when only a
    #               subset of the arguments have to be batched.
    # TODO(josipd): Do this smarter than n CustomCalls.
    print(f"batching {call_fn} with {args} and {batch_axes}")
    result = jax.lax.map(lambda x: call_fn(*x), args)
    print(f"batching gives result {result.shape} over axis {batch_axes}")
    print(
        "result has shape:",
    )
    for p in result:
        print("  ", p.shape)

    return result, batch_axes


def xla_register(name: str, fwd, bwd):
    # -- register custom c-funcs to XLA --
    name_fwd = name + "_forward"
    xla_client.register_custom_call_target(name_fwd, fwd, platform="gpu")
    name_bwd = name + "_backward"
    xla_client.register_custom_call_target(name_bwd, bwd, platform="gpu")
    return name_fwd,name_bwd

def cfunc_to_jax(name: str, fxn,
                 abstract_eval_fn, batching_fn=None):
    """Create a jittable JAX function for the given c-source function.
    Args:
      name: The name under which the primitive will be registered.
      gpu_fn: The c-source function
      abstract_eval_fn: The abstract evaluation function.
      batching_fn: If set, this function will be used when vmap-ing the returned
        function.
    Returns:
      A jitable JAX function.
    """

    # -- primitive --
    primitive = jax.core.Primitive(name)
    primitive.multiple_results = True
    # abstract_eval = partial(abstract_eval_rule, abstract_eval_fn)

    # -- define --
    primitive.def_abstract_eval(abstract_eval_fn)
    primitive.def_impl(partial(xla.apply_primitive, primitive))

    # -- register wrapped fwd/bwd (which use the registered c-funs) --
    xla.backend_specific_translations["gpu"][primitive] = fxn

    # -- skip batching for now --
    # if batching_fn is not None:
    #    batching.primitive_batchers[primitive] = batching_fn
    # batching.defvectorized(primitive)

    # xla.backend_specific_translations["cpu"][primitive] = partial(
    #     backends.cpu.xla_encode, gpu_fn, abstract_eval
    # )

    # -- return --
    return primitive

