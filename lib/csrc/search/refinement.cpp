#include <torch/extension.h>

#include <vector>

// CUDA forward declarations


void refinement_int_forward_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor qinds, torch::Tensor dists, torch::Tensor inds,
    int ws, int ps, int k, int stride0, int stride1, int dilation, int pt,
    bool restrict_radius, bool reflect_bounds, bool full_ws,
    int patch_offset, int dist_type);

void refinement_bilin2d_forward_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor qinds, torch::Tensor dists, torch::Tensor inds,
    int ws, int ps, int k, int stride0, int stride1, int dilation, int pt,
    bool restrict_radius, bool reflect_bounds, bool full_ws,
    int patch_offset, int dist_type);

void refinement_qinds_backward_cuda(
    torch::Tensor grad_qinds, const torch::Tensor grad_inds,
    const torch::Tensor qinds, const torch::Tensor inds);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void refinement_int_forward(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor qinds, torch::Tensor dists, torch::Tensor inds,
    int ws, int ps, int k, int stride0, int stride1, int dilation, int pt,
    bool restrict_radius, bool reflect_bounds, bool full_ws,
    int patch_offset, int dist_type){
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(qinds);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  refinement_int_forward_cuda(vid0, vid1, qinds, dists, inds,
                              ws, ps, k, stride0, stride1, dilation, pt,
                              restrict_radius, reflect_bounds, full_ws,
                              patch_offset, dist_type);
}

void refinement_bilin2d_forward(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor qinds, torch::Tensor dists, torch::Tensor inds,
    int ws, int ps, int k, int stride0, float stride1, int dilation, int pt,
    bool restrict_radius, bool reflect_bounds, bool full_ws,
    int patch_offset, int dist_type){
    // bool reflect_bounds, bool full_ws, int patch_offset, int dist_type){
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(qinds);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  refinement_bilin2d_forward_cuda(vid0, vid1, qinds, dists, inds,
                                  ws, ps, k, stride0, stride1, dilation, pt,
                                  restrict_radius, reflect_bounds, full_ws,
                                  patch_offset, dist_type);
}

void refinement_qinds_backward(
    torch::Tensor grad_qinds, const torch::Tensor grad_inds,
    const torch::Tensor qinds, const torch::Tensor inds){
  CHECK_INPUT(grad_qinds);
  CHECK_INPUT(grad_inds);
  CHECK_INPUT(qinds);
  CHECK_INPUT(inds);
  refinement_qinds_backward_cuda(grad_qinds, grad_inds, qinds, inds);

}

// python bindings
void init_refinement(py::module &m){
  m.def("refinement_int_forward", &refinement_int_forward,
        "Product Refine Forward (CUDA)");
  m.def("refinement_bilin2d_forward", &refinement_bilin2d_forward,
        "Product Refine Bilin2d Forward (CUDA)");
  m.def("refinement_qinds_backward", &refinement_qinds_backward,
        "Product Refine Backwards Indices (CUDA)");
  // m.def("ref_bwd_dists", &ref_bwd_dists,
  //       "Product Refine Backwards Dists (CUDA)");
}
