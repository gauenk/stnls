// [dir of col2im/im2col]: /home/gauenk/pytorch/aten/src/ATen/native/cuda/

// imports
#include <torch/extension.h>
// #include <torch/types.h>
#include <vector>
// #include "pybind.hpp"


// CUDA forward declarations

void dnls_cuda_ifoldz_forward(
    torch::Tensor vid, torch::Tensor zvid,
    torch::Tensor patches,
    int top, int left, int btm, int right,
    int start, int stride, int dilation, int adj,
    bool only_full, bool use_reflect);

// void dnls_cuda_ifoldz_backward(
//     torch::Tensor grad_vid,
//     torch::Tensor patches,
//     int top, int left, int btm, int right,
//     int start, int stride, int dilation, int adj,
//     bool only_full, bool use_reflect);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/*********************************

      Using Raster Order

*********************************/

void dnls_ifoldz_forward(
    torch::Tensor vid, torch::Tensor zvid,
    torch::Tensor patches,
    int top, int left, int btm, int right,
    int start, int stride, int dilation, int adj,
    bool only_full, bool use_reflect) {
  CHECK_INPUT(vid);
  CHECK_INPUT(zvid);
  CHECK_INPUT(patches);
  dnls_cuda_ifoldz_forward(vid,zvid,patches,
                           top,left,btm,right,
                           start,stride,dilation,adj,
                           only_full,use_reflect);
}

// void dnls_ifoldz_backward(
//     torch::Tensor grad_vid,
//     torch::Tensor patches,
//     int top, int left, int btm, int right,
//     int start, int stride, int dilation, int adj,
//     bool only_full, bool use_reflect) {
//   CHECK_INPUT(grad_vid);
//   CHECK_INPUT(patches);
//   dnls_cuda_ifoldz_backward(grad_vid,patches,
//                            top,left,btm,right,
//                            start,stride,dilation,adj,
//                            only_full,use_reflect);
// }



// python bindings
void init_ifoldz(py::module &m){
  m.def("ifoldz_forward", &dnls_ifoldz_forward, "DNLS Fold Forward (CUDA)");
  // m.def("ifoldz_backward", &dnls_ifoldz_backward, "DNLS Fold Backward (CUDA)");
}
