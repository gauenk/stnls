import torch as th

class ApproxSpaceSearchFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx,vid0,vid1,qshift=0,Q=-1):
        pass

    @staticmethod
    def backward(ctx):
        pass


class ApproxSpaceSearch(th.nn.Module):


    def __init__(self, ws, ps, k):
        super().__init__()
        pass

    def forward(self,vid0,vid1,qshift=0,nqueries=-1):
        return ApproxSpaceSearchFunction.apply(vid0,vid1,qshift,nqueries)

_apply = ApproxSpaceSearchFunction.apply # api
