#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void iwpsum_int_forward_cuda(
  torch::Tensor vid,
  torch::Tensor out_vid, torch::Tensor out_vidz,
  torch::Tensor dists, torch::Tensor inds,
  int ps, int pt, int dilation,
  bool reflect_bounds, bool use_adj);

void iwpsum_bilin2d_forward_cuda(
  torch::Tensor vid,
  torch::Tensor out_vid, torch::Tensor out_vidz,
  torch::Tensor dists, torch::Tensor inds,
  int ps, int pt, int dilation,
  bool reflect_bounds, bool use_adj);

void iwpsum_backward_vid_cuda(
    torch::Tensor vid_grad, torch::Tensor vid2fill_grad,
    torch::Tensor dists, torch::Tensor inds,
    int ps, int pt, int dilation,
    bool reflect_bounds, bool use_adj);

void iwpsum_backward_dists_cuda(
    torch::Tensor dists_grad, torch::Tensor in_grad,
    torch::Tensor vid, torch::Tensor inds,
    int ps, int pt, int dilation,
    bool reflect_bounds, bool use_adj);

void iwpsum_backward_inds_cuda(
    torch::Tensor dists_grad, torch::Tensor in_grad,
    torch::Tensor vid, torch::Tensor inds,
    int ps, int pt, int dilation,
    bool reflect_bounds, bool use_adj);


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void iwpsum_int_forward(
  torch::Tensor in_vid,
  torch::Tensor out_vid, torch::Tensor out_vidz,
  torch::Tensor dists, torch::Tensor inds,
  int ps, int pt, int dilation,
  bool reflect_bounds, bool use_adj){
  CHECK_INPUT(in_vid);
  CHECK_INPUT(out_vid);
  CHECK_INPUT(out_vidz);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  iwpsum_int_forward_cuda(in_vid,out_vid,out_vidz,dists,inds,
                      ps,pt,dilation,reflect_bounds,use_adj);
}

void iwpsum_bilin2d_forward(
  torch::Tensor in_vid,
  torch::Tensor out_vid, torch::Tensor out_vidz,
  torch::Tensor dists, torch::Tensor inds,
  int ps, int pt, int dilation,
  bool reflect_bounds, bool use_adj){
  CHECK_INPUT(in_vid);
  CHECK_INPUT(out_vid);
  CHECK_INPUT(out_vidz);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  iwpsum_bilin2d_forward_cuda(in_vid,out_vid,out_vidz,dists,inds,
                              ps,pt,dilation,reflect_bounds,use_adj);
}

void iwpsum_backward_vid(
  torch::Tensor out_grad, torch::Tensor in_grad,
  torch::Tensor dists, torch::Tensor inds,
  int ps, int pt, int dilation,
  bool reflect_bounds, bool use_adj){
  CHECK_INPUT(out_grad);
  CHECK_INPUT(in_grad);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  iwpsum_backward_vid_cuda(out_grad,in_grad,dists,inds,
                           ps,pt,dilation,reflect_bounds,use_adj);
}

void iwpsum_backward_dists(
  torch::Tensor dists_grad, torch::Tensor in_grad,
  torch::Tensor vid, torch::Tensor inds,
  int ps, int pt, int dilation, bool reflect_bounds, bool use_adj){
  CHECK_INPUT(dists_grad);
  CHECK_INPUT(in_grad);
  CHECK_INPUT(vid);
  CHECK_INPUT(inds);
  iwpsum_backward_dists_cuda(dists_grad,in_grad,vid,inds,
                             ps,pt,dilation,reflect_bounds,use_adj);
}

void iwpsum_backward_inds(
  torch::Tensor dists_grad, torch::Tensor in_grad,
  torch::Tensor vid, torch::Tensor inds,
  int ps, int pt, int dilation, bool reflect_bounds, bool use_adj){
  CHECK_INPUT(dists_grad);
  CHECK_INPUT(in_grad);
  CHECK_INPUT(vid);
  CHECK_INPUT(inds);
  // iwpsum_backward_dists_cuda(dists_grad,in_grad,vid,inds,
  //                            ps,pt,dilation,reflect_bounds,use_adj);
}


// python bindings
void init_iwpsum(py::module &m){
  m.def("iwpsum_int_forward", &iwpsum_int_forward,
        "(Vid) WeightedSum Forward (CUDA)");
  m.def("iwpsum_bilin2d_forward", &iwpsum_bilin2d_forward,
        "(Vid) WeighedSum Forward (CUDA)");
  m.def("iwpsum_backward_vid", &iwpsum_backward_vid,
        "(Bid) WeightedPatchSum Backward (CUDA)");
  m.def("iwpsum_backward_dists", &iwpsum_backward_dists,
        "(Vid) WeightedPatchSum Backward (CUDA)");
}

