#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void cuda_wpsum_heads_forward(
  torch::Tensor vid, torch::Tensor patches,
  torch::Tensor dists, torch::Tensor inds,
  int h_off, int w_off,
  int dilation, int adj, bool reflect_bounds);

void cuda_wpsum_heads_backward_vid(
    torch::Tensor vid_grad, torch::Tensor patches_grad,
    torch::Tensor dists, torch::Tensor inds,
    int h_off, int w_off,
    int dilation, int adj, bool reflect_bounds, bool exact);

void cuda_wpsum_heads_backward_dists(
    torch::Tensor dists_grad, torch::Tensor patches_grad,
    torch::Tensor vid, torch::Tensor inds,
    int h_off, int w_off,
    int dilation, int adj, bool reflect_bounds, bool exact);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void wpsum_heads_forward(
  torch::Tensor vid, torch::Tensor patches,
  torch::Tensor dists, torch::Tensor inds,
  int h_off, int w_off,
  int dilation, int adj, bool reflect_bounds){
  CHECK_INPUT(vid);
  CHECK_INPUT(patches);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  cuda_wpsum_heads_forward(vid,patches,dists,inds,
                          h_off,w_off,dilation,adj,reflect_bounds);
}

void wpsum_heads_backward_vid(
  torch::Tensor vid_grad, torch::Tensor patches_grad,
  torch::Tensor dists, torch::Tensor inds,
  int h_off, int w_off, int dilation,
  int adj, bool reflect_bounds, bool exact){
  CHECK_INPUT(vid_grad);
  CHECK_INPUT(patches_grad);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  cuda_wpsum_heads_backward_vid(vid_grad,patches_grad,dists,inds,
                           h_off,w_off,dilation,adj,reflect_bounds,exact);
}

void wpsum_heads_backward_dists(
  torch::Tensor dists_grad, torch::Tensor patches_grad,
  torch::Tensor vid, torch::Tensor inds,
  int h_off, int w_off, int dilation,
  int adj, bool reflect_bounds, bool exact){
  CHECK_INPUT(dists_grad);
  CHECK_INPUT(patches_grad);
  CHECK_INPUT(vid);
  CHECK_INPUT(inds);
  cuda_wpsum_heads_backward_dists(dists_grad,patches_grad,vid,inds,
                                 h_off,w_off,dilation,adj,reflect_bounds,exact);
}


// python bindings
void init_wpsum_heads(py::module &m){
  m.def("wpsum_heads_forward", &wpsum_heads_forward,"DNLS WeightedPatchSumHeads Forward (CUDA)");
  m.def("wpsum_heads_backward_vid", &wpsum_heads_backward_vid,"DNLS WeightedPatchSumHeads Backward (CUDA)");
  m.def("wpsum_heads_backward_dists", &wpsum_heads_backward_dists,"DNLS WeightedPatchSumHeads Backward (CUDA)");
}

