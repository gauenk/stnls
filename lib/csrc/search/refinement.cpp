#include <torch/extension.h>

#include <vector>

// CUDA forward declarations


void refinement_forward_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor qinds, torch::Tensor dists, torch::Tensor inds,
    int ws_h, int ws_w, int ps, int k, int dist_type, int stride0, int stride1,
    int dilation, int pt, int qshift, bool reflect_bounds, bool full_ws,
    bool use_adj, int off_H0, int off_W0, int off_H1, int off_W1);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void refinement_forward(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor qinds, torch::Tensor dists, torch::Tensor inds,
    int ws_h, int ws_w, int ps, int k, int dist_type, int stride0, int stride1,
    int dilation, int pt, int qshift, bool reflect_bounds, bool full_ws,
    bool use_adj, int off_H0, int off_W0, int off_H1, int off_W1){
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(qinds);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  refinement_forward_cuda(vid0, vid1, qinds, dists, inds,
                          ws_h, ws_w, ps, k, dist_type, stride0, stride1,
                          dilation, pt, qshift, reflect_bounds, full_ws,
                          use_adj, off_H0, off_W0, off_H1, off_W1);
}

// python bindings
void init_refinement(py::module &m){
  m.def("refinement_forward", &refinement_forward,
        "Product Refine Forward (CUDA)");
}
