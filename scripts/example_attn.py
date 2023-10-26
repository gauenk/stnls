# -- imports --
import torch as th
conv2d = th.nn.functional.conv2d
conv3d = th.nn.functional.conv3d
from einops import rearrange
import stnls

# -- init --
B,T = 1,5 # batch size, number of frames
F,H,W = 16,128,128 # number of features, height, width
device = "cuda"
V_in = th.randn((B,T,F,H,W),device=device)
vshape = V_in.shape
V_in = V_in.requires_grad_(True)

# -- transform --
proj_weights = th.randn((F,F,1,1),device=device)
q_vid = conv2d(V_in.view(-1,*vshape[2:]),proj_weights).view(vshape)
k_vid = conv2d(V_in.view(-1,*vshape[2:]),proj_weights).view(vshape)
v_vid = conv2d(V_in.view(-1,*vshape[2:]),proj_weights).view(vshape)

# -- search info --
ws = 5 # spatial window size
wt = 2 # temporal window size; searching total frames W_t = 2*wt+1
ps,K,HD = 3,10,2 # patch size, number of neighbors, number of heads
stride0,stride1 = 1,0.5 # query & key stride

# -- accumulate optical flows --
fflow = th.randn((B,T,2,H,W),device=device)
bflow = th.randn((B,T,2,H,W),device=device)
flows = stnls.nn.search_flow(fflow,bflow,wt,stride0)
# print(flows.shape) (B,T,W_t-1,2,H,W); W_t = 2*wt+1

# -- search --
search = stnls.search.NonLocalSearch(ws,wt,ps,K,nheads=HD,
                                     stride0=stride0,stride1=stride1,
                                     self_action="anchor",itype="float")
dists,srch_flows = search(q_vid,k_vid,flows)
# print(srch_flows.shape) # B,HD,T,nH,nW,K,3; nH=(H-1)//stride0+1

# -- normalize --
weights = th.nn.functional.softmax(10*dists,-1)

# -- aggregate --
ps = 5 # patch size can change for stacking
stack = stnls.agg.NonLocalStack(ps,stride0)
stacked = stack(v_vid,weights,srch_flows)
# stacked.shape = (B,HD,K,T,F',H,W) where F' = F/HD
V_out = rearrange(stacked,'b hd k t f h w -> (b t) (hd f) k h w')
proj_weights = th.randn((F,F,K,1,1),device=device)
V_out = conv3d(V_out,proj_weights,stride=(K,1,1))
V_out = rearrange(V_out,'(b t) f 1 h w -> b t f h w',b=B)
print("V_out.shape: ",V_out.shape) # B,T,F,H,W

V_grad = th.randn_like(V_out)
th.autograd.backward(V_out,V_grad)
print(V_in.grad.shape)
