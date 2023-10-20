"""
Read nasty pybind11 .so names

"""
# /usr/lib/x86_64-linux-gnu/libcuda.so
# "/home/gauenk/Documents/packages/stnls/lib/stnls_cuda.cpython-38-x86_64-linux-gnu.so"

from lief import parse # read symbols
from pathlib import Path

_symbols = []
def stnls_symbols():
    root = Path(__file__).parents[0] / "../../"
    obj = root / "stnls_cuda.cpython-38-x86_64-linux-gnu.so"
    for sym in parse(obj).symbols:
        _symbols.append(sym)

def get_stnls_symbol(name):
    if len(_symbols) == 0:
        stnls_symbols()
    for sym in _symbols:
        pass


