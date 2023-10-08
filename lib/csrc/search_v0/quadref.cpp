#include <torch/extension.h>

#include <vector>

// CUDA forward declarations


void quadref_forward_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1,
    // const torch::Tensor deno0, const torch::Tensor deno1,
    const torch::Tensor qinds, torch::Tensor dists, torch::Tensor inds,
    int ws_h, int ws_w, int ps, int k, int dist_type, int stride0, int stride1,
    int dilation, int pt, int qshift, bool reflect_bounds, bool full_ws,
    bool use_adj, int off_H0, int off_W0, int off_H1, int off_W1);

void quadref_backward_cuda(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    torch::Tensor vid0, torch::Tensor vid1,
    // torch::Tensor deno0, torch::Tensor deno1,
    torch::Tensor grad_dists, torch::Tensor inds,
    int q_shift, int stride0, int nH0, int nW0,
    int ps, int pt, int dilation, bool reflect_bounds,
    bool use_adj, int off_H0, int off_W0, int off_H1, int off_W1,
    bool use_rand, bool exact, int dist_type,
    int queries_per_thread, int neigh_per_thread, int channel_groups);


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void quadref_forward(
    const torch::Tensor vid0, const torch::Tensor vid1,
    // const torch::Tensor deno0, const torch::Tensor deno1,
    const torch::Tensor qinds, torch::Tensor dists, torch::Tensor inds,
    int ws_h, int ws_w, int ps, int k, int dist_type, int stride0, int stride1,
    int dilation, int pt, int qshift, bool reflect_bounds, bool full_ws,
    bool use_adj, int off_H0, int off_W0, int off_H1, int off_W1){
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  // CHECK_INPUT(deno0);
  // CHECK_INPUT(deno1);
  CHECK_INPUT(qinds);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  quadref_forward_cuda(vid0, vid1,
                       // deno0, deno1,
                       qinds, dists, inds,
                       ws_h, ws_w, ps, k, dist_type, stride0, stride1,
                       dilation, pt, qshift, reflect_bounds, full_ws,
                       use_adj, off_H0, off_W0, off_H1, off_W1);
}

void quadref_backward(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    torch::Tensor vid0, torch::Tensor vid1,
    // torch::Tensor deno0, torch::Tensor deno1,
    torch::Tensor grad_dists, torch::Tensor inds,
    int q_shift, int stride0, int nH0, int nW0,
    int ps, int pt, int dilation, bool reflect_bounds,
    bool use_adj, int off_H0, int off_W0, int off_H1, int off_W1,
    bool use_rand, bool exact, int dist_type,
    int queries_per_thread, int neigh_per_thread, int channel_groups) {
  CHECK_INPUT(grad_vid0);
  CHECK_INPUT(grad_vid1);
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  // CHECK_INPUT(deno0);
  // CHECK_INPUT(deno1);
  CHECK_INPUT(grad_dists);
  CHECK_INPUT(inds);
  quadref_backward_cuda(grad_vid0, grad_vid1, vid0, vid1,
                        // deno0, deno1,
                        grad_dists, inds, q_shift, stride0, nH0, nW0,
                        ps, pt, dilation, reflect_bounds,
                        use_adj, off_H0, off_W0, off_H1, off_W1,
                        use_rand, exact, dist_type,
                        queries_per_thread, neigh_per_thread, channel_groups);
}


// python bindings
void init_quadref(py::module &m){
  m.def("quadref_forward", &quadref_forward,
        "Quadradic Refine Forward (CUDA)");
  m.def("quadref_backward", &quadref_backward,
        "Quadradic Refine Backward (CUDA)");
}
