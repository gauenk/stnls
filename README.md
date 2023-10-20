# stnls

A small CUDA library to support Space-Time Attention with a Shifted Non-Local Search.

## Summary


## Install

```bash
git clone git@github.com:gauenk/stnls.git
cd stnls
python -m pip install -e .
```

## Usage (torch.nn.Module)

See `scripts/example_attn.py`

```python

import torch as th
import stnls

# -- init videos -- 
B = 1 # batchsize
T = 3 # number of frames
C = 12 # number of channels; each head uses C//nheads
H = 128 # height
W = 128 # width
vid0 = th.randn((B,T,C,H,W),device="cuda:0",dtype=th.float32)
vid1 = th.randn((B,T,C,H,W),device="cuda:0",dtype=th.float32)

# -- init optical flow (forward and backward) --
fflow = th.randn((B,T,2,H,W),device="cuda:0",dtype=th.float32)
bflow = th.randn((B,T,2,H,W),device="cuda:0",dtype=th.float32)

# -- init layer --
ws = 21 # spatial window (ws) in num of pixels
wt = 3 # time window (wt) in one direction; 2*wt+1 total frames
ps = 7 # patch size
k = 10 # number of neighbors
nheads = 4 # number of heads (in attention). Splits channel dimension.
search_layer = stnls.search.NonLocalSearch(ws, wt, ps, k, nheads)

# -- run search --
dists,inds = search_layer(vid0,vid1,fflow,bflow)

# dists.shape = (B,nheads,Q,K)
# inds.shape = (B,nheads,Q,K,3)
# Q = number of patches searched = T * ((H-1)//stride0+1) * ((W-1)//stride0+1)
```

## Usage (Functional)

```python

import torch as th
import stnls

# -- init --
B,T,C,H,W = 4,3,12,128,128
vid0 = th.randn((B,T,C,H,W),device="cuda:0",dtype=th.float32)
vid1 = th.randn((B,T,C,H,W),device="cuda:0",dtype=th.float32)
fflow = th.randn((B,T,2,H,W),device="cuda:0",dtype=th.float32)
bflow = th.randn((B,T,2,H,W),device="cuda:0",dtype=th.float32)
ws = 21 # spatial window (ws) in num of pixels
wt = 3 # time window (wt) in one direction; 2*wt+1 total frames
ps = 7 # patch size
k = 10 # number of neighbors
nheads = 4 # number of heads (in attention). Splits channel dimension.

# -- run search --
search_layer = stnls.search.nls(vid0, vid1, fflow, bflow, ws, wt, ps, k, nheads)

```

