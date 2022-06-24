## Related Code

In this section, we distinguish this code-based from similar code:

[NAT](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer): This paper proposes using a neighborhood window for the attention map, rather than the entire image. A core CUDA kernel, [linked here](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer/blob/main/natten/src/nattenqkrpb_cuda_kernel.cu), efficiently computes a neighborhood dot-product between Q and V. The `dnls` code base's `search`
function is similar, but we compute patch similarity (i) using an optional optical flow
and (ii) using the L2-norm instead of the dot product.
 
[N3Net](https://github.com/visinf/n3net): This paper proposes a differentiable non-local K nearest neighbors search. Core CUDA kernels are [linked here](https://github.com/visinf/n3net/blob/master/lib/matmul1_kernel.cu) and [here](https://github.com/visinf/n3net/blob/master/lib/matmul1_bwd_kernel.cu). These kernels allow for efficient multiplication using indices in 1d. However, this kernel is only used for testing. For training, they compute the nearest neighbors search by first expanding the entire search space of features, say 1,000 - 5,000 sets of features, for each search location in the batch. This duplicatation of data consumes large amounts of GPU memory. 

[pyinn](https://github.com/szagoruyko/pyinn): This project combines Cupy and Pytorch functions. A core CUDA kernel, [linked here](https://github.com/szagoruyko/pyinn/blob/948388e4ee585b23ed41d352fc8863ea868874ad/pyinn/im2col.py#L48), computes the fold and unfold (im2col and col2im) functions. Their code does not allow for batching, so it is limited in the same way as standard fold and unfold.
