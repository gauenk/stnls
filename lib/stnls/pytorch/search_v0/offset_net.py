"""

A standard network to compute offsets for NonLocalSearchOffsets

"""


import torch as th
import torch.nn as nn

def fwarp(vid,flow,st):
    pass

def bwarp(vid,flow,st):
    pass

def imap(inds):
    pass

class OffsetNet(nn.Module):

    def __init__(self,nftrs,ws,wt,
                 nheads=1,ps=3,stride0=1,pt=1,use_search=False,dist_type="l2"):
        self.ws = ws
        self.nftrs = nftrs
        self.use_search = use_search
        self.net = nn.Sequential(*[
            nn.Conv2d(nftrs,nftrs,kernel_size=(3,3),padding="same")
        ])
        self.search = None
        if self.use_search:
            self.search = NonLocalSearch(ws,wt,ps,k,nheads,
                                         dist_type=dist_type,pt=pt,stride0=stride0)

    def get_ftrs(self,vid0,vid1,fflow,bflow):
        if self.use_search:
            dists,inds = search(vid0,vid1,fflow,bflow)
            stack = stack(vid1,dists,inds)
            imap = inds2map(inds)
            ftrs = th.cat([vid0,stack,imap],-3)
        else:
            ST = 2*self.wt+1
            stack = []
            for st in range(ST):
                stack.append(fwarp(vid1,fflow,st))
                stack.append(bwarp(vid1,bflow,st))
            stack = th.cat(stack,-3)
            ftrs = th.cat([vid0,stack,fflow,bflow],-3)
        return ftrs

    def forward(self,vid0,vid1,fflow,bflow):
        ftrs = self.get_ftrs(vid0,vid1,fflow,bflow)
        offsets = self.net(ftrs)
        offsets = self.normalize(offsets)
        return ofests

