"""
Copied from https://github.com/PhilipVinc/numba4jax/
"""

import jax

from jax.abstract_arrays import ShapedArray

from .c_api import pycapsule_new


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


def default_primitive_name(fun) -> str:
    return f"njit4jax[{fun.__module__}.{fun.__name__}]"

#@title Helper functions (execute this cell)
import functools
import traceback

_indentation = 0
def _trace(msg=None):
    """Print a message at current indentation."""
    if msg is not None:
        print("  " * _indentation + msg)

def _trace_indent(msg=None):
    """Print a message and then indent the rest."""
    global _indentation
    _trace(msg)
    _indentation = 1 + _indentation

def _trace_unindent(msg=None):
    """Unindent then print a message."""
    global _indentation
    _indentation = _indentation - 1
    _trace(msg)

def trace(name):
  """A decorator for functions to trace arguments and results."""

  def trace_func(func):  # pylint: disable=missing-docstring
    def pp(v):
        """Print certain values more succinctly"""
        vtype = str(type(v))
        if "jax._src.lib.xla_bridge._JaxComputationBuilder" in vtype:
            return "<JaxComputationBuilder>"
        elif "jaxlib.xla_extension.XlaOp" in vtype:
            return "<XlaOp at 0x{:x}>".format(id(v))
        elif ("partial_eval.JaxprTracer" in vtype or
              "batching.BatchTracer" in vtype or
              "ad.JVPTracer" in vtype):
            return "Traced<{}>".format(v.aval)
        elif isinstance(v, tuple):
            return "({})".format(pp_values(v))
        else:
            return str(v)
    def pp_values(args):
        return ", ".join([pp(arg) for arg in args])

    @functools.wraps(func)
    def func_wrapper(*args):
      _trace_indent("call {}({})".format(name, pp_values(args)))
      res = func(*args)
      _trace_unindent("|<- {} = {}".format(name, pp(res)))
      return res

    return func_wrapper

  return trace_func

class expectNotImplementedError(object):
  """Context manager to check for NotImplementedError."""
  def __enter__(self): pass
  def __exit__(self, type, value, tb):
    global _indentation
    _indentation = 0
    if type is NotImplementedError:
      print("\nFound expected exception:")
      traceback.print_exc(limit=3)
      return True
    elif type is None:  # No exception
      assert False, "Expected NotImplementedError"
    else:
      return False
