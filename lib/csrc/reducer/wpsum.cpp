#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void wpsum_forward_cuda(
  torch::Tensor vid, torch::Tensor patches,
  torch::Tensor dists, torch::Tensor inds,
  int h_off, int w_off,
  int dilation, bool use_adj, bool reflect_bounds);

void wpsum_backward_vid_cuda(
    torch::Tensor vid_grad, torch::Tensor patches_grad,
    torch::Tensor dists, torch::Tensor inds,
    int h_off, int w_off,
    int dilation, bool use_adj, bool reflect_bounds,
    bool use_rand, bool exact, bool use_atomic);

void wpsum_backward_dists_cuda(
    torch::Tensor dists_grad, torch::Tensor patches_grad,
    torch::Tensor vid, torch::Tensor inds,
    int h_off, int w_off,
    int dilation, bool use_adj, bool reflect_bounds,
    bool exact, bool use_atomic);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void wpsum_forward(
  torch::Tensor vid, torch::Tensor patches,
  torch::Tensor dists, torch::Tensor inds,
  int h_off, int w_off,
  int dilation, bool use_adj, bool reflect_bounds){
  CHECK_INPUT(vid);
  CHECK_INPUT(patches);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  wpsum_forward_cuda(vid,patches,dists,inds,
		     h_off,w_off,dilation,use_adj,reflect_bounds);
}

void wpsum_backward_vid(
  torch::Tensor vid_grad, torch::Tensor patches_grad,
  torch::Tensor dists, torch::Tensor inds,
  int h_off, int w_off, int dilation,
  bool use_adj, bool reflect_bounds, bool use_rand,
  bool exact, bool use_atomic){
  CHECK_INPUT(vid_grad);
  CHECK_INPUT(patches_grad);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  wpsum_backward_vid_cuda(vid_grad,patches_grad,dists,inds,
			  h_off,w_off,dilation,use_adj,reflect_bounds,
			  use_rand,exact,use_atomic);
}

void wpsum_backward_dists(
  torch::Tensor dists_grad, torch::Tensor patches_grad,
  torch::Tensor vid, torch::Tensor inds,
  int h_off, int w_off, int dilation,
  bool use_adj, bool reflect_bounds, bool exact, bool use_atomic){
  CHECK_INPUT(dists_grad);
  CHECK_INPUT(patches_grad);
  CHECK_INPUT(vid);
  CHECK_INPUT(inds);
  wpsum_backward_dists_cuda(dists_grad,patches_grad,vid,inds,
			    h_off,w_off,dilation,use_adj,reflect_bounds,
			    exact, use_atomic);
}


// python bindings
void init_wpsum(py::module &m){
  m.def("wpsum_forward", &wpsum_forward,"WeightedPatchSum Forward (CUDA)");
  m.def("wpsum_backward_vid",&wpsum_backward_vid,"WeightedPatchSum Backward (CUDA)");
  m.def("wpsum_backward_dists",&wpsum_backward_dists,"WeightedPatchSum Backward (CUDA)");
}

