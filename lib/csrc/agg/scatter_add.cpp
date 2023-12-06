// imports
#include <torch/extension.h>
// #include <torch/types.h>
#include <vector>
// #include "pybind.hpp"


// CUDA forward declarations

void scatter_add_int_forward_cuda(
  torch::Tensor out_vid, torch::Tensor counts,
  const torch::Tensor in_vid,
  const torch::Tensor dists, const torch::Tensor inds,
  int ps, int strideIn, int strideOut, int pt,
  int dilation, bool reflect_bounds, int patch_offset);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/*********************************

      Using Raster Order

*********************************/


void scatter_add_int_forward(
  torch::Tensor out_vid, torch::Tensor counts,
  const torch::Tensor in_vid,
  const torch::Tensor dists,
  const torch::Tensor inds,
  int ps, int strideIn, int strideOut, int pt,
  int dilation, bool reflect_bounds, int patch_offset){
  CHECK_INPUT(out_vid);
  CHECK_INPUT(counts);
  CHECK_INPUT(in_vid);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  scatter_add_int_forward_cuda(out_vid,counts,in_vid,dists,inds,
                               ps,strideIn,strideOut,pt,dilation,
                               reflect_bounds,patch_offset);
}


// void scatter_add_int_backward(
//     torch::Tensor out_grad, const torch::Tensor in_grad,
//     const torch::Tensor vid, const torch::Tensor weights,
//     const torch::Tensor inds,  const torch::Tensor labels,
//     torch::Tensor stack, torch::Tensor mask, torch::Tensor counts,
//     int ps, int pt, int dilation, int stride0, bool reflect_bounds, int patch_offset){
//   CHECK_INPUT(vid);
//   CHECK_INPUT(weights);
//   CHECK_INPUT(inds);
//   CHECK_INPUT(labels);
//   CHECK_INPUT(stack);
//   CHECK_INPUT(mask);
//   CHECK_INPUT(counts);
//   scatter_add_int_backward_cuda(vid,weights,inds,labels,stack,mask,counts,
//                                    ps,pt,dilation,stride0,
//                                    reflect_bounds,patch_offset);
// }

// -- python bindings --
void init_scatter_add(py::module &m){
  m.def("scatter_add_int_forward",
        &scatter_add_int_forward,
        "Scatter Forward with Int Indexing");
  m.def("scatter_add_int_backward",
        &scatter_add_int_backward,
        "Scatter Backward with Int Indexing");
}
