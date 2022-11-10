from . import pfc


def init(version,*args,**kwargs):
    if version == "pfc":
        return pfc.PatchFC(*args,**kwargs)
    else:
        raise ValueError(f"Uknown version [{version}]")

