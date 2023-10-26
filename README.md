# Space-Time Non-Local Search (stnls)

A Pytorch-friendly C++/CUDA library to support Space-Time Attention with a Shifted Non-Local Search. The shifted non-local search corrects the small spatial inaccuracies from predicted, long-range offsets such as optical flow.

[arxiv](https://arxiv.org/pdf/2309.16849.pdf)

![related works](https://github.com/gauenk/stnls/blob/master/figs/compare_search.png?raw=true)
![shifted nls](https://github.com/gauenk/stnls/blob/master/figs/shifted_nls.png?raw=true)


## Install & Usage

```bash
git clone git@github.com:gauenk/stnls.git
cd stnls
python -m pip install -e .
```

See "example_attn.py" for usage details. Another example is below:

```
import torch as th
import stnls

# -- init --
B,T = 1,5 # batch size, number of frames
F,H,W = 16,128,128 # number of features, height, width
device = "cuda"
q_vid = th.randn((B,T,F,H,W),device=device)
k_vid = th.randn((B,T,F,H,W),device=device)
v_vid = th.randn((B,T,F,H,W),device=device)

# -- search info --
ws = 5 # spatial window size
wt = 2 # temporal window size; searching total frames W_t = 2*wt+1
ps,K,HD = 3,10,2 # patch size, number of neighbors, number of heads
stride0,stride1 = 1,0.5 # query & key stride

# -- run search --
search = stnls.search.NonLocalSearch(ws,wt,ps,K,nheads=HD,
                                     stride0=stride0,stride1=stride1,
                                     self_action="anchor",itype="float")
dists,srch_flows = search(q_vid,k_vid,flows)
# print(srch_flows.shape) # B,HD,T,nH,nW,K,3; nH=(H-1)//stride0+1

# -- normalize --
weights = th.nn.functional.softmax(10*dists,-1)

# -- aggregate --
agg = stnls.agg.WeightedPatchSum(ps=ps,stride0=stride0,itype="float")
V_out = agg(v_vid,weights,srch_flows)
print("V_out.shape: ",V_out.shape) # B,T,F,H,W
```


## Experiments

### Alignment Results

![shifted nls](https://github.com/gauenk/stnls/blob/master/figs/align_grid.png?raw=true)



