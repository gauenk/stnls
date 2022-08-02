#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void cuda_wpsum_heads_2vid_forward(
  torch::Tensor vid, torch::Tensor vid_fill,
  torch::Tensor dists, torch::Tensor inds,
  int h_off, int w_off,
  int qstart, int ps, int pt,
  int stride, int dilation, int adj, bool reflect_bounds, bool only_full);

void cuda_wpsum_heads_2vid_backward_vid(
    torch::Tensor vid_grad, torch::Tensor vid_fill_grad,
    torch::Tensor dists, torch::Tensor inds,
    int h_off, int w_off,
    int qstart, int ps, int pt,
    int stride, int dilation, int adj,
    bool reflect_bounds, bool only_full, bool exact);

void cuda_wpsum_heads_2vid_backward_dists(
    torch::Tensor dists_grad, torch::Tensor vid_fill_grad,
    torch::Tensor vid, torch::Tensor inds,
    int h_off, int w_off,
    int qstart, int ps, int pt,
    int stride, int dilation, int adj,
    bool reflect_bounds, bool only_full, bool exact);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void wpsum_heads_2vid_forward(
  torch::Tensor vid, torch::Tensor vid_fill,
  torch::Tensor dists, torch::Tensor inds,
  int h_off, int w_off,
  int qstart, int ps, int pt, int stride, int dilation,
  int adj, bool reflect_bounds, bool only_full){
  CHECK_INPUT(vid);
  CHECK_INPUT(vid_fill);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  cuda_wpsum_heads_2vid_forward(vid,vid_fill,dists,inds,h_off,w_off,
                                qstart,
                                ps, pt, stride, dilation, adj,
                                reflect_bounds, only_full);
}

void wpsum_heads_2vid_backward_vid(
  torch::Tensor vid_grad, torch::Tensor vid_fill_grad,
  torch::Tensor dists, torch::Tensor inds,
  int h_off, int w_off, int qstart,
  int ps, int pt, int stride, int dilation, int adj,
  bool reflect_bounds, bool only_full, bool exact){
  CHECK_INPUT(vid_grad);
  CHECK_INPUT(vid_fill_grad);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  cuda_wpsum_heads_2vid_backward_vid(vid_grad,vid_fill_grad,
                                     dists,inds,
                                     h_off,w_off,
                                     qstart,ps,pt,stride,dilation,adj,
                                     reflect_bounds,only_full,exact);
}

void wpsum_heads_2vid_backward_dists(
  torch::Tensor dists_grad, torch::Tensor vid_fill_grad,
  torch::Tensor vid, torch::Tensor inds,
  int h_off, int w_off,
  int qstart,
  int ps, int pt, int stride, int dilation,
  int adj, bool reflect_bounds, bool only_full, bool exact){
  CHECK_INPUT(dists_grad);
  CHECK_INPUT(vid_fill_grad);
  CHECK_INPUT(vid);
  CHECK_INPUT(inds);
  cuda_wpsum_heads_2vid_backward_dists(dists_grad,vid_fill_grad,
                                       vid,inds,
                                       h_off,w_off,
                                       qstart,ps,pt,stride,dilation,adj,
                                       reflect_bounds,only_full,exact);
}


// python bindings
void init_wpsum_heads_2vid(py::module &m){
  m.def("wpsum_heads_2vid_forward", &wpsum_heads_2vid_forward,
        "DNLS WeightedPatchSumHeads Forward (CUDA)");
  m.def("wpsum_heads_2vid_backward_vid", &wpsum_heads_2vid_backward_vid,
        "DNLS WeightedPatchSumHeads Backward (CUDA)");
  m.def("wpsum_heads_2vid_backward_dists", &wpsum_heads_2vid_backward_dists,
        "DNLS WeightedPatchSumHeads Backward (CUDA)");
}

