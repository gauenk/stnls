"""
Read nasty pybind11 .so names

"""
# /usr/lib/x86_64-linux-gnu/libcuda.so
# "/home/gauenk/Documents/packages/dnls/lib/dnls_cuda.cpython-38-x86_64-linux-gnu.so"

from lief import parse # read symbols
from pathlib import Path

_symbols = []
def dnls_symbols():
    root = Path(__file__).parents[0] / "../../"
    obj = root / "dnls_cuda.cpython-38-x86_64-linux-gnu.so"
    for sym in parse(obj).symbols:
        _symbols.append(sym)

def get_dnls_symbol(name):
    if len(_symbols) == 0:
        dnls_symbols()
    for sym in _symbols:
        pass


