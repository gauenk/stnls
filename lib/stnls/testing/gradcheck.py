"""

   Example:

   my_fxn = lambda vid: stnls.nn.another_fxn(vid)
   ana = stnls.testing.gradcheck.get_ana_jacobian(my_fxn,vid)
   num = stnls.testing.gradcheck.get_num_jacobian(my_fxn,vid,eps=1e-2,nreps=1)
   assert th.abs(ana-num).mean()<1e-3

"""

import torch

def get_num_jacobian(fxn,inputs,eps=1e-3,nreps=1):
    from torch.autograd.gradcheck import _get_numerical_jacobian
    num = _get_numerical_jacobian(fxn, (inputs,),
                                  eps=eps, is_forward_ad=False)[0][0]
    for i in range(nreps-1):
        num += get_num_jacobian(fxn,inputs,eps=eps)
    num /= nreps
    return num

def get_ana_jacobian(fxn,inputs,eps=1e-5):
    from torch.autograd.gradcheck import _check_analytical_jacobian_attributes
    out = fxn(inputs)
    ana = _check_analytical_jacobian_attributes((inputs,), out, eps, False)[0]
    return ana

def get_gradcheck_pair(fxn,inputs,eps=1e-3):
    num = get_num_jacobian(fxn,inputs,eps=1e-3)
    ana = get_ana_jacobian(fxn,inputs)
    return num,ana
