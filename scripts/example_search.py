"""

Example Search Script

"""

# -- imports --
import torch as th
conv2d = th.nn.functional.conv2d
conv3d = th.nn.functional.conv3d
from einops import rearrange
import stnls

# -- init --
B = 1 # batch size
T = 5 # number of frames
F = 16 # number of features
H = 128 # image height
W = 128 # image width
device = "cuda"
V_in = th.randn((B,T,F,H,W),device=device)
vshape = V_in.shape

# -- transform --
proj_weights = th.randn((F,F,1,1),device=device)
q_vid = conv2d(V_in.view(-1,*vshape[2:]),proj_weights).view(vshape)
k_vid = conv2d(V_in.view(-1,*vshape[2:]),proj_weights).view(vshape)
v_vid = conv2d(V_in.view(-1,*vshape[2:]),proj_weights).view(vshape)

# -- search info --
ws = 5 # spatial window size
wt = 2 # temporal window size; searching total frames W_t = 2*wt
ps = 3 # patch size
K = 10 # number of neighbors
nheads = 2 # number of heads
stride0 = 1 # query stride
stride1 = 1 # key stride

# -- accumulate optical flows --
fflow = th.randn((B,T,2,H,W),device=device)
bflow = th.randn((B,T,2,H,W),device=device)
flows = stnls.nn.search_flow(fflow,bflow,wt,stride0)
# print(flows.shape) (B,T,W_t,2,H,W); W_t = 2*wt

# -- search --
search = stnls.search.NonLocalSearch(ws,wt,ps,K,nheads,
                                     stride0=stride0,stride1=stride1,
                                     self_action="anchor",itype="float")
dists,inds = search(q_vid,k_vid,flows)
# print(inds.shape) # B,HD,T,nH,nW,K,3

# -- normalize --
weights = th.nn.functional.softmax(10*dists,-1)

# -- aggregate --
ps = 5 # patch size can change for stacking
stack = stnls.agg.NonLocalGather(ps,stride0)
V_out = stack(v_vid,weights,inds)
V_out = rearrange(V_out,'b hd k t f h w -> (b t) (hd f) k h w')
proj_weights = th.randn((F,F,K,1,1),device=device)
V_out = conv3d(V_out,proj_weights,stride=(K,1,1))
V_out = rearrange(V_out,'(b t) f 1 h w -> b t f h w',b=B)
# print("V_out.shape: ",V_out.shape) # B,T,F,H,W


