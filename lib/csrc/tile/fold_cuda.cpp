// [dir of col2im/im2col]: /home/gauenk/pytorch/aten/src/ATen/native/cuda/

// imports
#include <torch/extension.h>
// #include <torch/types.h>
#include <vector>
// #include "pybind.hpp"


// CUDA forward declarations

void stnls_cuda_fold_forward(
    torch::Tensor vid,
    torch::Tensor patches,
    int start, int stride,
    int adj, int dilation);

void stnls_cuda_fold_backward(
    torch::Tensor grad_vid,
    torch::Tensor patches,
    int start, int stride,
    int adj, int dilation);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/*********************************

      Using Raster Ordering

*********************************/

void stnls_fold_forward(
    torch::Tensor vid,
    torch::Tensor patches,
    int start, int stride,
    int adj, int dilation) {
  CHECK_INPUT(vid);
  CHECK_INPUT(patches);
  stnls_cuda_fold_forward(vid,patches,start,stride,adj,dilation);
}

void stnls_fold_backward(
    torch::Tensor grad_vid,
    torch::Tensor patches,
    int start, int stride,
    int adj, int dilation) {
  CHECK_INPUT(grad_vid);
  CHECK_INPUT(patches);
  stnls_cuda_fold_backward(grad_vid,patches,start,stride,adj,dilation);
}


// python bindings
void init_fold(py::module &m){
  m.def("fold_forward", &stnls_fold_forward, "DNLS Fold Forward (CUDA)");
  m.def("fold_backward", &stnls_fold_backward, "DNLS Fold Backward (CUDA)");
}

