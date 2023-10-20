#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void wpsum_int_forward_cuda(
  torch::Tensor vid,
  torch::Tensor out_vid, torch::Tensor out_vidz,
  torch::Tensor dists, torch::Tensor inds,
  int ps, int pt, int dilation,
  bool reflect_bounds, bool use_adj);

void wpsum_int_backward_vid_cuda(
    torch::Tensor vid_grad, torch::Tensor vid2fill_grad,
    torch::Tensor dists, torch::Tensor inds,
    int ps, int pt, int dilation,
    bool reflect_bounds, bool use_adj);

void wpsum_int_backward_dists_cuda(
    torch::Tensor dists_grad, torch::Tensor in_grad,
    torch::Tensor vid, torch::Tensor inds,
    int ps, int pt, int dilation,
    bool reflect_bounds, bool use_adj);

void wpsum_bilin2d_forward_cuda(
  torch::Tensor vid,
  torch::Tensor out_vid, torch::Tensor out_vidz,
  torch::Tensor dists, torch::Tensor inds,
  int ps, int pt, int dilation,
  bool reflect_bounds, bool use_adj);

void wpsum_bilin2d_backward_vid_cuda(
    torch::Tensor vid_grad, torch::Tensor vid2fill_grad,
    torch::Tensor dists, torch::Tensor inds,
    int ps, int pt, int dilation,
    bool reflect_bounds, bool use_adj);

void wpsum_bilin2d_backward_dists_cuda(
    torch::Tensor dists_grad, torch::Tensor in_grad,
    torch::Tensor vid, torch::Tensor inds,
    int ps, int pt, int dilation,
    bool reflect_bounds, bool use_adj);

void wpsum_bilin2d_backward_inds_cuda(
    torch::Tensor dists_grad, torch::Tensor in_grad,
    torch::Tensor vid, torch::Tensor inds,
    int ps, int pt, int dilation,
    bool reflect_bounds, bool use_adj);


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/***********************


      Int Indexing


***********************/

void wpsum_int_forward(
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
  wpsum_int_forward_cuda(in_vid,out_vid,out_vidz,dists,inds,
                      ps,pt,dilation,reflect_bounds,use_adj);
}

void wpsum_int_backward_vid(
  torch::Tensor out_grad, torch::Tensor in_grad,
  torch::Tensor dists, torch::Tensor inds,
  int ps, int pt, int dilation,
  bool reflect_bounds, bool use_adj){
  CHECK_INPUT(out_grad);
  CHECK_INPUT(in_grad);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  wpsum_backward_vid_cuda(out_grad,in_grad,dists,inds,
                           ps,pt,dilation,reflect_bounds,use_adj);
}

void wpsum_int_backward_dists(
  torch::Tensor dists_grad, torch::Tensor in_grad,
  torch::Tensor vid, torch::Tensor inds,
  int ps, int pt, int dilation, bool reflect_bounds, bool use_adj){
  CHECK_INPUT(dists_grad);
  CHECK_INPUT(in_grad);
  CHECK_INPUT(vid);
  CHECK_INPUT(inds);
  wpsum_backward_dists_cuda(dists_grad,in_grad,vid,inds,
                             ps,pt,dilation,reflect_bounds,use_adj);
}

/***********************


      Bilinear2d


***********************/

void wpsum_bilin2d_forward(
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
  wpsum_bilin2d_forward_cuda(in_vid,out_vid,out_vidz,dists,inds,
                              ps,pt,dilation,reflect_bounds,use_adj);
}


/***********************


    Python Bindings


***********************/

void init_wpsum(py::module &m){
  m.def("wpsum_int_forward", &wpsum_int_forward,
        "WeightedPatchSum Forward (CUDA)");
  m.def("wpsum_int_backward_vid", &wpsum_int_backward_vid,
        "WeightedPatchSum Backward (CUDA)");
  m.def("wpsum_int_backward_dists", &wpsum_int_backward_dists,
        "WeightedPatchSum Backward (CUDA)");
  m.def("wpsum_bilin2d_forward", &wpsum_bilin2d_forward,
        "WeightedPatchSum Forward (CUDA)");
  m.def("wpsum_bilin2d_backward_vid", &wpsum_bilin2d_backward_vid,
        "WeightedPatchSum Backward (CUDA)");
  m.def("wpsum_bilin2d_backward_dists", &wpsum_bilin2d_backward_dists,
        "WeightedPatchSum Backward (CUDA)");
  m.def("wpsum_bilin2d_backward_inds", &wpsum_bilin2d_backward_inds,
        "WeightedPatchSum Backward (CUDA)");

}

