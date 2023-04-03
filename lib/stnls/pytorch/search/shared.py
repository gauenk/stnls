"""

Shared Logical Units

"""

import dnls

def manage_self(dists,inds,anchor_self,remove_self,qshift,stride0,H,W):
    assert not(remove_self and anchor_self)
    if remove_self:
        outs = dnls.nn.remove_self(dists,inds,stride0,H,W,qhift)
        dists,inds = outs
    if anchor_self:
        dnls.nn.anchor_self(dists,inds,stride0,H,W,qshift)
    return dists,inds


