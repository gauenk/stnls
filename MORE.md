## Patch-based Processing: Graph Neural Networks, Attention, and Non-Local Denoising

We would like to be able to operate on image patches, rather than the entire image. 
This is motivated by research such as [Graph Neural Networks](https://arxiv.org/abs/1812.08434), [VIT](https://arxiv.org/pdf/2010.11929.pdf), [NAT](https://arxiv.org/abs/2204.07143), and [LIDIA](https://arxiv.org/pdf/1911.07167.pdf).
Operating on image patches allows patch information to be transformed independently from it's neighbor.
Another motivation for operating on patches comes from operating on tokens from natural language processing.

Since I am studying image denoising, I note non-local denoising methods are a type of [transformer](https://openreview.net/pdf?id=MmujBClawFo) and also a [graph neural networks](https://arxiv.org/abs/1905.12281) (since [transformers are a special case of graph nerual networks](https://graphdeeplearning.github.io/post/transformers-are-gnns/)). 
These operations often look like the following code block,

```python
patches = unfold(video)
patches_mod = model(patches)
vide_mod = fold(patches_mod)
```

Runnning `unfold` on the entire video at once
requires tons of GPU memory.
This code base provides differentiable, patch-based, 
batch-friendly (or video friendly) CUDA operations 
within Pytorch to place a cap on the memory requirement
using the following pseudo-code,


```python
nbatches = (npixels-1)//batch_size + 1
for batch in range(nbatches):
    patch_batch = unfold(video,batch)
    patch_batch_mod = model(patch_batch)
    video_mod += fold(patch_batch_mod,batch)
```
