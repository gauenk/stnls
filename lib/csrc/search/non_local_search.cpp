#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void non_local_search_forward_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor fflow, const torch::Tensor bflow,
    torch::Tensor dists, torch::Tensor inds,
    int wt, int ps, int k, int dist_type,
    int stride0, int stride1, int dilation, int pt, int qshift,
    bool reflect_bounds, bool full_ws, bool search_abs,
    bool use_adj, int off_H0, int off_W0, int off_H1, int off_W1);

void non_local_search_backward_cuda(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor grad_dists, torch::Tensor inds,
    int q_shift, int stride0, int nH0, int nW0,
    int ps, int pt, int dilation, bool reflect_bounds,
    bool use_adj, int off_H0, int off_W0, int off_H1, int off_W1,
    bool use_rand, bool exact, int dist_type,
    int queries_per_thread, int neigh_per_thread, int channel_groups);

// void non_local_search_backward_cuda(
//     torch::Tensor grad_vid0, torch::Tensor grad_vid1,
//     torch::Tensor vid0, torch::Tensor vid1,
//     torch::Tensor dists, torch::Tensor inds,
//     int qshift, int stride0, int n_h0, int n_w0,
//     int off_H0, int off_W0, int off_H1, int off_W1,
//     int ps, int pt, int dilation, bool use_adj,
//     bool reflect_bounds, bool use_rand, bool exact, int dist_type);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void non_local_search_forward(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor fflow, const torch::Tensor bflow,
    torch::Tensor dists, torch::Tensor inds,
    int wt, int ps, int k, int dist_type,
    int stride0, int stride1, int dilation, int pt, int qshift,
    bool reflect_bounds, bool full_ws, bool search_abs, bool use_adj,
    int off_H0, int off_W0, int off_H1, int off_W1){
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(fflow);
  CHECK_INPUT(bflow);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  non_local_search_forward_cuda(vid0, vid1, fflow, bflow, dists, inds,
                                wt, ps, k, dist_type,
                                stride0, stride1, dilation, pt, qshift,
                                reflect_bounds, full_ws, search_abs,
                                use_adj, off_H0, off_W0, off_H1, off_W1);
}

void non_local_search_backward(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor grad_dists, torch::Tensor inds,
    int q_shift, int stride0, int nH0, int nW0,
    int ps, int pt, int dilation, bool reflect_bounds,
    bool use_adj, int off_H0, int off_W0, int off_H1, int off_W1,
    bool use_rand, bool exact, int dist_type,
    int channel_groups, int neigh_per_thread, int queries_per_thread) {
  CHECK_INPUT(grad_vid0);
  CHECK_INPUT(grad_vid1);
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(grad_dists);
  CHECK_INPUT(inds);
  non_local_search_backward_cuda(grad_vid0, grad_vid1, vid0, vid1,
                                 grad_dists, inds, q_shift, stride0, nH0, nW0,
                                 ps, pt, dilation, reflect_bounds,
                                 use_adj, off_H0, off_W0, off_H1, off_W1,
                                 use_rand, exact, dist_type,
                                 queries_per_thread, neigh_per_thread, channel_groups);
}


// python bindings
void init_non_local_search(py::module &m){
  m.def("non_local_search_forward", &non_local_search_forward,
        "Search Forward with Heads (CUDA)");
  m.def("non_local_search_backward", &non_local_search_backward,
        "Search Backward with Heads (CUDA)");
}

