#include <torch/extension.h>

#include <vector>

// CUDA forward declarations


void refinement_int_forward_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1, const torch::Tensor flows,
    torch::Tensor dists, torch::Tensor inds,
    int ws, int ps, int stride0, int stride1, int dilation, int pt,
    bool restrict_radius, bool reflect_bounds, bool full_ws,
    int patch_offset, int off_Hq, int off_Wq, int dist_type);

void refinement_bilin2d_forward_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1, const torch::Tensor flows,
    torch::Tensor dists, torch::Tensor inds,
    torch::Tensor kselect, torch::Tensor reflect,
    int ws, int ps, int stride0, float stride1, int dilation, int pt,
    bool restrict_radius, bool reflect_bounds, bool full_ws,
    int patch_offset, int off_Hq, int off_Wq, int dist_type);

void refinement_bilin2d_vidflows_backward_cuda(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1, torch::Tensor grad_flows,
    const torch::Tensor vid0, const torch::Tensor vid1,// const torch::Tensor flows,
    const torch::Tensor grad_dists, const torch::Tensor grad_inds,
    // const torch::Tensor dists,
    const torch::Tensor inds,
    const torch::Tensor kselect, const torch::Tensor reflect,
    int wt, int ps, int pt, int stride0, int dilation,
    bool reflect_bounds, int patch_offset, int off_Hq, int off_Wq, int dist_type);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void refinement_int_forward(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor flows, torch::Tensor dists, torch::Tensor inds,
    int ws, int ps, int stride0, int stride1, int dilation, int pt,
    bool restrict_radius, bool reflect_bounds, bool full_ws,
    int patch_offset, int off_Hq, int off_Wq, int dist_type){
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(flows);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  refinement_int_forward_cuda(vid0, vid1, flows, dists, inds,
                              ws, ps, stride0, stride1, dilation, pt,
                              restrict_radius, reflect_bounds, full_ws,
                              patch_offset, off_Hq, off_Wq, dist_type);
}

void refinement_bilin2d_forward(
    const torch::Tensor vid0, const torch::Tensor vid1, const torch::Tensor flows,
    torch::Tensor dists, torch::Tensor inds,
    torch::Tensor kselect, torch::Tensor reflect,
    int ws, int ps, int stride0, float stride1, int dilation, int pt,
    bool restrict_radius, bool reflect_bounds, bool full_ws,
    int patch_offset, int off_Hq, int off_Wq, int dist_type){
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(flows);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  CHECK_INPUT(kselect);
  CHECK_INPUT(reflect);
  refinement_bilin2d_forward_cuda(vid0, vid1, flows, dists, inds,
                                  kselect, reflect,
                                  ws, ps, stride0, stride1, dilation, pt,
                                  restrict_radius, reflect_bounds, full_ws,
                                  patch_offset, off_Hq, off_Wq, dist_type);
}

void refinement_bilin2d_vidflows_backward(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1, torch::Tensor grad_flows,
    const torch::Tensor vid0, const torch::Tensor vid1, //const torch::Tensor flows,
    const torch::Tensor grad_dists, const torch::Tensor grad_inds,
    // const torch::Tensor dists,
    const torch::Tensor inds,
    const torch::Tensor kselect, const torch::Tensor reflect,
    int wt, int ps, int pt, int stride0, int dilation,
    bool reflect_bounds, int patch_offset,
    int off_Hq, int off_Wq, int dist_type) {
  CHECK_INPUT(grad_vid0);
  CHECK_INPUT(grad_vid1);
  CHECK_INPUT(grad_flows);
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  // CHECK_INPUT(flows);
  CHECK_INPUT(grad_dists);
  CHECK_INPUT(grad_inds);
  // CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  CHECK_INPUT(kselect);
  CHECK_INPUT(reflect);
  refinement_bilin2d_vidflows_backward_cuda(grad_vid0, grad_vid1, grad_flows,
                                            vid0, vid1, //flows,
                                            grad_dists, grad_inds,
                                            // dists,
                                            inds, kselect, reflect,
                                            wt, ps, pt, stride0, dilation,
                                            reflect_bounds, patch_offset,
                                            off_Hq, off_Wq, dist_type);
}

// python bindings
void init_refinement(py::module &m){
  m.def("refinement_int_forward", &refinement_int_forward,
        "Non-Local Search Surrounding K Flows with Int Indexing (CUDA)");
  m.def("refinement_bilin2d_forward", &refinement_bilin2d_forward,
        "Non-Local Search Surrounding K Flows with Float Indexing (CUDA)");
  m.def("refinement_bilin2d_vidflows_backward",
        &refinement_bilin2d_vidflows_backward,
        "Refine Backwards Indices (CUDA)");
}
