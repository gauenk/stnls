from . import pfc
from . import optical_flow_accumulate as ofa

def init(version,*args,**kwargs):
    if version == "pfc":
        return pfc.PatchFC(*args,**kwargs)
    elif version == "ofa":
        return ofa.init(*args)
    else:
        raise ValueError(f"Uknown version [{version}]")


def run(version,*args,**kwargs):
    if version == "ofa":
        return ofa.run(*args)
    else:
        raise ValueError(f"Uknown version [{version}]")
