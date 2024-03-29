// imports
#include <torch/extension.h>
// #include <torch/types.h>
#include <vector>
// #include "pybind.hpp"


// CUDA forward declarations

void scatter_int_forward_cuda(
    const torch::Tensor vid, const torch::Tensor weights,
    const torch::Tensor inds, const torch::Tensor labels,
    torch::Tensor stack, torch::Tensor mask, torch::Tensor counts,
    int ps, int pt, int dilation, int stride0,
    bool reflect_bounds, int patch_offset);

// void scatter_tensor_forward_cuda(torch::Tensor out_tensor,
//                                  const torch::Tensor in_tensor,
//                                  const torch::Tensor labels,
//                                  const torch::Tensor flows_k,
//                                  int stride0, int stride1, int H, int W);

// void scatter_tensor_backward_cuda(torch::Tensor in_tensor_grad,
//                                   const torch::Tensor out_tensor_grad,
//                                   const torch::Tensor labels,
//                                   const torch::Tensor flows_k, int stride0);

// void scatter_bilin2d_forward_cuda(
//     const torch::Tensor vid, const torch::Tensor weights,
//     const torch::Tensor inds, torch::Tensor stack, torch::Tensor counts,
//     int ps, int pt, int dilation, int stride0,
//     bool reflect_bounds, int patch_offset);

// void scatter_int_backward_cuda(
//     torch::Tensor grad_vid,
//     torch::Tensor grad_weights,
//     const torch::Tensor grad_stack,
//     const torch::Tensor vid,
//     const torch::Tensor weights,
//     const torch::Tensor inds,
//     const torch::Tensor stack,
//     const torch::Tensor counts,
//     int ps, int pt, int dilation, int stride0,
//     bool reflect_bounds, int patch_offset);

// void scatter_bilin2d_backward_cuda(
//     torch::Tensor grad_vid,
//     torch::Tensor grad_weights,
//     torch::Tensor grad_inds,
//     const torch::Tensor grad_stack,
//     const torch::Tensor vid,
//     const torch::Tensor weights,
//     const torch::Tensor inds,
//     const torch::Tensor stack,
//     const torch::Tensor counts,
//     int ps, int pt, int dilation, int stride0,
//     bool reflect_bounds, int patch_offset);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void scatter_int_forward(
    const torch::Tensor vid, const torch::Tensor weights,
    const torch::Tensor inds,  const torch::Tensor labels,
    torch::Tensor stack, torch::Tensor mask, torch::Tensor counts,
    int ps, int pt, int dilation, int stride0, bool reflect_bounds, int patch_offset){
  CHECK_INPUT(vid);
  CHECK_INPUT(weights);
  CHECK_INPUT(inds);
  CHECK_INPUT(labels);
  CHECK_INPUT(stack);
  CHECK_INPUT(mask);
  CHECK_INPUT(counts);
  scatter_int_forward_cuda(vid,weights,inds,labels,stack,mask,counts,
                           ps,pt,dilation,stride0,
                           reflect_bounds,patch_offset);
}

// void scatter_bilin2d_forward(
//     const torch::Tensor vid, const torch::Tensor weights,
//     const torch::Tensor inds, torch::Tensor stack, torch::Tensor counts,
//     int ps, int pt, int dilation, int stride0, bool reflect_bounds, int patch_offset){
//   CHECK_INPUT(vid);
//   CHECK_INPUT(weights);
//   CHECK_INPUT(inds);
//   CHECK_INPUT(stack);
//   CHECK_INPUT(counts);

//     scatter_bilin2d_forward_cuda(vid,weights,inds,stack,counts,
//                                          ps,pt,dilation,stride0,
//                                          reflect_bounds,patch_offset);
// }


// void scatter_int_backward(
//     torch::Tensor grad_vid,
//     torch::Tensor grad_weights,
//     const torch::Tensor grad_stack,
//     const torch::Tensor vid,
//     const torch::Tensor weights,
//     const torch::Tensor inds,
//     const torch::Tensor stack,
//     const torch::Tensor counts,
//     int ps, int pt, int dilation, int stride0,
//     bool reflect_bounds, int patch_offset){
//   CHECK_INPUT(grad_vid);
//   CHECK_INPUT(grad_weights);
//   CHECK_INPUT(grad_stack);
//   CHECK_INPUT(vid);
//   CHECK_INPUT(weights);
//   CHECK_INPUT(inds);
//   CHECK_INPUT(stack);
//   CHECK_INPUT(counts);
//   scatter_int_backward_cuda(grad_vid,grad_weights,grad_stack,
//                                     vid,weights,inds,stack,counts,
//                                     ps,pt,dilation,stride0,
//                                     reflect_bounds,patch_offset);
// }


// void scatter_bilin2d_backward(
//     torch::Tensor grad_vid,
//     torch::Tensor grad_weights,
//     torch::Tensor grad_inds,
//     const torch::Tensor grad_stack,
//     const torch::Tensor vid,
//     const torch::Tensor weights,
//     const torch::Tensor inds,
//     const torch::Tensor stack,
//     const torch::Tensor counts,
//     int ps, int pt, int dilation, int stride0,
//     bool reflect_bounds, int patch_offset){
//   CHECK_INPUT(grad_vid);
//   CHECK_INPUT(grad_weights);
//   CHECK_INPUT(grad_inds);
//   CHECK_INPUT(grad_stack);
//   CHECK_INPUT(vid);
//   CHECK_INPUT(weights);
//   CHECK_INPUT(inds);
//   CHECK_INPUT(stack);
//   CHECK_INPUT(counts);
//   scatter_bilin2d_backward_cuda(grad_vid,grad_weights,grad_inds,grad_stack,
//                                         vid,weights,inds,stack,counts,
//                                         ps,pt,dilation,stride0,
//                                         reflect_bounds,patch_offset);
// }


// python bindings
void init_scatter(py::module &m){
  // m.def("scatter_labels", &scatter_labels,
  //       "Scatter Labels");
  // m.def("scatter_tensor_forward", &scatter_tensor_forward,
  //       "Scatter Tensor");
  // m.def("scatter_tensor_backward", &scatter_tensor_backward,
  //       "Scatter Tensor");
  m.def("scatter_int_forward", &scatter_int_forward,
        "Scatter Forward with Int Indexing");
  // m.def("scatter_int_backward",&scatter_int_backward,
  //       "Scatter Backward with Int Indexing");
}

