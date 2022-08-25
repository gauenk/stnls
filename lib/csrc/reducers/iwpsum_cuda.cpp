#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void cuda_iwpsum_forward(
  torch::Tensor vid, torch::Tensor vid2fill,
  torch::Tensor dists, torch::Tensor inds,
    int ps, int pt, int h_off, int w_off,
  int dilation, int adj, bool reflect_bounds);

void cuda_iwpsum_backward_vid(
    torch::Tensor vid_grad, torch::Tensor vid2fill_grad,
    torch::Tensor dists, torch::Tensor inds,
    int ps, int pt, int h_off, int w_off,
    int dilation, int adj, bool reflect_bounds, bool exact);

void cuda_iwpsum_backward_dists(
    torch::Tensor dists_grad, torch::Tensor vid2fill_grad,
    torch::Tensor vid, torch::Tensor inds,
    int ps, int pt, int h_off, int w_off,
    int dilation, int adj, bool reflect_bounds, bool exact);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void iwpsum_forward(
  torch::Tensor vid, torch::Tensor vid2fill,
  torch::Tensor dists, torch::Tensor inds,
  int ps, int pt, int h_off, int w_off,
  int dilation, int adj, bool reflect_bounds){
  CHECK_INPUT(vid);
  CHECK_INPUT(vid2fill);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  cuda_iwpsum_forward(vid,vid2fill,dists,inds,
                      ps,pt,h_off,w_off,dilation,
                      adj,reflect_bounds);
}

void iwpsum_backward_vid(
  torch::Tensor vid_grad, torch::Tensor vid2fill_grad,
  torch::Tensor dists, torch::Tensor inds,
  int ps, int pt, int h_off, int w_off,
  int dilation, int adj, bool reflect_bounds, bool exact){
  CHECK_INPUT(vid_grad);
  CHECK_INPUT(vid2fill_grad);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  cuda_iwpsum_backward_vid(vid_grad,vid2fill_grad,dists,inds,
                           ps,pt,h_off,w_off,dilation,adj,
                           reflect_bounds,exact);
}

void iwpsum_backward_dists(
  torch::Tensor dists_grad, torch::Tensor vid2fill_grad,
  torch::Tensor vid, torch::Tensor inds,
  int ps, int pt, int h_off, int w_off,
  int dilation, int adj, bool reflect_bounds, bool exact){
  CHECK_INPUT(dists_grad);
  CHECK_INPUT(vid2fill_grad);
  CHECK_INPUT(vid);
  CHECK_INPUT(inds);
  cuda_iwpsum_backward_dists(dists_grad,vid2fill_grad,vid,inds,
                             ps,pt,h_off,w_off,dilation,adj,
                             reflect_bounds,exact);
}


// python bindings
void init_iwpsum(py::module &m){
  m.def("iwpsum_forward", &iwpsum_forward,
        "DNLS In-Place WeightedPatchSum Forward (CUDA)");
  m.def("iwpsum_backward_vid", &iwpsum_backward_vid,
        "DNLS In-Place WeightedPatchSum Backward (CUDA)");
  m.def("iwpsum_backward_dists", &iwpsum_backward_dists,
        "DNLS In-Place WeightedPatchSum Backward (CUDA)");
}

