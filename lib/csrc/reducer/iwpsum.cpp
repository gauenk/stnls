#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void iwpsum_forward_cuda(
  torch::Tensor vid, torch::Tensor vid2fill,
  torch::Tensor dists, torch::Tensor inds,
    int ps, int pt, int h_off, int w_off,
  int dilation, int adj, bool reflect_bounds);

void iwpsum_backward_vid_cuda(
    torch::Tensor vid_grad, torch::Tensor vid2fill_grad,
    torch::Tensor dists, torch::Tensor inds,
    int ps, int pt, int h_off, int w_off,
    int dilation, int adj, bool reflect_bounds, bool exact);

void iwpsum_backward_dists_cuda(
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
  iwpsum_forward_cuda(vid,vid2fill,dists,inds,
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
  iwpsum_backward_vid_cuda(vid_grad,vid2fill_grad,dists,inds,
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
  iwpsum_backward_dists_cuda(dists_grad,vid2fill_grad,vid,inds,
                             ps,pt,h_off,w_off,dilation,adj,
                             reflect_bounds,exact);
}


// python bindings
void init_iwpsum(py::module &m){
  m.def("iwpsum_forward", &iwpsum_forward,
        "(Vid) WeightedPatchSum Forward (CUDA)");
  m.def("iwpsum_backward_vid", &iwpsum_backward_vid,
        "(Bid) WeightedPatchSum Backward (CUDA)");
  m.def("iwpsum_backward_dists", &iwpsum_backward_dists,
        "(Vid) WeightedPatchSum Backward (CUDA)");
}

