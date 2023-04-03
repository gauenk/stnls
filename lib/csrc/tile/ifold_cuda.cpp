// [dir of col2im/im2col]: /home/gauenk/pytorch/aten/src/ATen/native/cuda/

// imports
#include <torch/extension.h>
// #include <torch/types.h>
#include <vector>
// #include "pybind.hpp"


// CUDA forward declarations

void stnls_cuda_ifold_forward(
    torch::Tensor vid,
    torch::Tensor patches,
    int top, int left, int btm, int right,
    int start, int stride, int dilation, int adj,
    bool only_full, bool use_reflect);

void stnls_cuda_ifold_backward(
    torch::Tensor grad_vid,
    torch::Tensor patches,
    int top, int left, int btm, int right,
    int start, int stride, int dilation, int adj,
    bool only_full, bool use_reflect);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/*********************************

      Using Raster Order

*********************************/

void stnls_ifold_forward(
    torch::Tensor vid,
    torch::Tensor patches,
    int top, int left, int btm, int right,
    int start, int stride, int dilation, int adj,
    bool only_full, bool use_reflect) {
  CHECK_INPUT(vid);
  CHECK_INPUT(patches);
  stnls_cuda_ifold_forward(vid,patches,
                          top,left,btm,right,
                          start,stride,dilation,adj,
                          only_full,use_reflect);
}

void stnls_ifold_backward(
    torch::Tensor grad_vid,
    torch::Tensor patches,
    int top, int left, int btm, int right,
    int start, int stride, int dilation, int adj,
    bool only_full, bool use_reflect) {
  CHECK_INPUT(grad_vid);
  CHECK_INPUT(patches);
  stnls_cuda_ifold_backward(grad_vid,patches,
                           top,left,btm,right,
                           start,stride,dilation,adj,
                           only_full,use_reflect);
}



// python bindings
void init_ifold(py::module &m){
  m.def("ifold_forward", &stnls_ifold_forward, "DNLS Fold Forward (CUDA)");
  m.def("ifold_backward", &stnls_ifold_backward, "DNLS Fold Backward (CUDA)");
}

