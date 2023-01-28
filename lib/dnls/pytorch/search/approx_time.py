import torch as th

class ApproxTimeSearchFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx,vid0,vid1,qshift=0,Q=-1):
        pass

    @staticmethod
    def backward(ctx):
        pass


class ApproxTimeSearch(th.nn.Module):


    def __init__(self, ws, ps, k):
        super().__init__()
        pass

    def forward(self,vid0,vid1,qshift=0,nqueries=-1):
        return ApproxTimeSearchFunction.apply(vid0,vid1,qshift,nqueries)

_apply = ApproxTimeSearchFunction.apply # api
