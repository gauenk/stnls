"""

Shared Logical Units

"""

import stnls

def manage_self(dists,inds,anchor_self,remove_self,qshift,stride0,H,W):
    assert not(remove_self and anchor_self)
    if remove_self:
        outs = stnls.nn.remove_self(dists,inds,stride0,H,W,qhift)
        dists,inds = outs
    if anchor_self:
        stnls.nn.anchor_self(dists,inds,stride0,H,W,qshift)
    return dists,inds


