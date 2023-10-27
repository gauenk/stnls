"""

   A consolidated locations of a few projection operators for
   aggregation within attention to sysmetically access them.

"""

import torch.nn as nn
from stnls.utils import optional
from stnls.utils import extract_pairs


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#         Initialize the Projection Version
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_config(_cfg,restrict=True):
    version = optional(_cfg,"nlstack_proj_version","v1")
    defaults = get_defaults(version)
    defaults["nlstack_proj_version"] = version
    cfg = extract_pairs(_cfg,defaults,restrict=restrict)
    # cfg.nlstack_proj_version = nlstack_proj_version
    return cfg

def init(cfg):
    cfg = extract_config(cfg,False)
    return get_projection(cfg)

def get_defaults(version):
    if version == "v1":
        return {"ps":-1,"embed_dim":-1,"inner_mult":-1,"k_agg":-1,"nheads":-1,
                "attn_drop_rate_proj":0.}
    elif version == "v2":
        return {"attn_proj_ksize":-1,"attn_proj_stride":"k_ps_ps",
                "attn_proj_ngroups":"ngroups","attn_drop_rate_proj":0.}
    else:
        raise ValueError(f"Uknown version for projection menu [{version}]")

def get_projection(cfg):
    proj,proj_drop = None,None
    io_dim = cfg.embed_dim * cfg.nheads
    version = cfg.nlstack_proj_version
    if version == "v1":
        ps = 3#search_cfg.ps
        proj = nn.Conv3d(io_dim*cfg.inner_mult,io_dim,
                         kernel_size=(cfg.k_agg,cfg.ps,cfg.ps),
                         stride=(cfg.k_agg,1,1),
                         padding=(0,cfg.ps//2,cfg.ps//2),
                         groups=cfg.nheads)
        proj_drop = nn.Dropout(cfg.attn_drop_rate_proj)
    elif version == "v2":
        if "_" in cfg.attn_proj_ksize:
            ksizes = []
            for ksize_str in cfg.attn_proj_ksize.split("_"):
                if ksize_str == "k": ksizes.append(kagg)
                elif ksize_str == "ps": ksizes.append(self.ps)
                elif ksize_str == "ps//2": ksizes.append(self.ps//2)
                else: ksizes.append(int(ksize_str))
        else:
            msg = "Uknown proj kernel size. [%s]"
            raise ValueError(msg % cfg.attn_proj_ksize)
        pads = [0,ksizes[1]//2,ksizes[2]//2]

        if "_" in cfg.attn_proj_stride:
            strides = []
            for stride_str in cfg.attn_proj_stride.split("_"):
                if stride_str == "k": strides.append(kagg)
                elif stride_str == "ps": strides.append(self.ps)
                elif stride_str == "ps//2": strides.append(self.ps//2)
                else: strides.append(int(stride_str))
        else:
            msg = "Uknown proj kernel size. [%s]"
            raise ValueError(msg % cfg.attn_proj_ksize)

        if cfg.attn_proj_ngroups == "nheads":
            ngroups = search_cfg.nheads
        elif isinstance(cfg.attn_proj_ngroups,int):
            ngroups = cfg.attn_proj_ngroups
        else:
            raise ValueError("Uknown proj ngroups [%s]" % cfg.attn_proj_ngroups)

        proj = nn.Conv3d(io_dim*inner_mult,io_dim,
                              kernel_size=ksizes,stride=strides,
                              padding=pads,groups=ngroups)
        proj_drop = nn.Dropout(cfg.attn_drop_rate_proj)
    else:
        raise NotImplementedError("")
    return proj,proj_drop


