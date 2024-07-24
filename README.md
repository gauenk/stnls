# Space-Time Non-Local Search (stnls)

A Pytorch-friendly C++/CUDA library to support Space-Time Attention with a Shifted Non-Local Search. The shifted non-local search corrects the small spatial inaccuracies from predicted, long-range offsets such as optical flow (as in Guided Deformable Attention).

[[arxiv](https://arxiv.org/pdf/2309.16849.pdf)]

## Related Works & Module Summary

Our module corrects small spatial errors of long-range predicted offsets to identify regions of high-affinity between the query and keys within attention. The module executes a small grid search surrounding the predicted offset locations. The first figure compares our search method with other recent attention modules. The second figure outlines each conceptual step of our Shifted Non-Local Search.

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


## Snippet of Results 

### Video Alignment

The Shifted Non-Local Search (Shifted-NLS) corrects the small spatial errors of predicted offsets such as optical flow. This section illustrates the significant impact of these small spatial errors through video alignment. This experiment uses the first 10 frames from the DAVIS training dataset. When searching and computing the TV-L1 optical flow, we add a small amount of Gaussian noise (σ = 15) to simulate the uncertainty of the trained query and key values of an attention module within a network during training

![shifted nls](https://github.com/gauenk/stnls/blob/master/figs/align_grid.png?raw=true)

### Upgrading Existing Space-Time Attention

We upgrade Guided Deformable Attention (GDA) with our Shifted Non-Local Search (Shifted-NLS) module to show the value of correcting the errors of predicted offsets for video denoising [rvrt](https://github.com/JingyunLiang/RVRT). GDA requires 9 offsets for each pixel in the image. In the original network, 9 offsets are output from a small convolution network whose input includes optical flow. Our method omits the small network and searches the local region surrounding the optical flow. In this experiment, our spatial window is 9x9 and the temporal window is fixed to 1 by architecture design. The most similar 9 locations are selected to replace the offsets from the network. Table 1 shows the denoising quality improves when using our search method compared to using predicted offsets. The improvement is between 0.20 - 0.40 dB across all noise levels, an increase often attributed to an entirely new architecture.

![upgrading rvrt](https://github.com/gauenk/stnls/blob/master/figs/upgrade_rvrt.png?raw=true)


## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{gauen2023space,
  title={Space-Time Attention with Shifted Non-Local Search},
  author={Gauen, Kent and Chan, Stanley},
  journal={arXiv},
  year={2023}
}
```

