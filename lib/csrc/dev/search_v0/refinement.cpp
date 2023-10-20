#include <torch/extension.h>

#include <vector>

// CUDA forward declarations


void refinement_forward_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor qinds, torch::Tensor dists, torch::Tensor inds,
    int ws_h, int ws_w, int ps, int k, int dist_type, int stride0, int stride1,
    int dilation, int pt, int qshift, bool reflect_bounds, bool full_ws,
    bool use_adj, int off_H0, int off_W0, int off_H1, int off_W1);

void refinement_forward_bilin2d_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor qinds, torch::Tensor dists, torch::Tensor inds,
    int ws_h, int ws_w, int ps, int k, int dist_type, int stride0, int stride1,
    int dilation, int pt, int qshift, bool reflect_bounds, bool full_ws,
    bool use_adj, int off_H0, int off_W0, int off_H1, int off_W1);

void non_local_search_backward_cuda(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor grad_dists, const torch::Tensor inds,
    int q_shift, int stride0, int nH0, int nW0,
    int ps, int pt, int dilation, bool reflect_bounds,
    bool use_adj, int off_H0, int off_W0,
    int off_H1, int off_W1, int dist_type);

void ref_bwd_dists_bilin2d_cuda(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor grad_dists, const torch::Tensor inds,
    int q_shift, int stride0, int nH0, int nW0,
    int ps, int pt, int dilation, bool reflect_bounds,
    bool use_adj, int off_H0, int off_W0,
    int off_H1, int off_W1, int dist_type);

void ref_bwd_inds_cuda(
    torch::Tensor grad_qinds, const torch::Tensor grad_inds,
    const torch::Tensor qinds, const torch::Tensor inds);


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

void refinement_bilin2d_forward(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor qinds, torch::Tensor dists, torch::Tensor inds,
    int ws_h, int ws_w, int ps, int k, int dist_type, int stride0, float stride1,
    int dilation, int pt, int qshift, bool reflect_bounds, bool full_ws,
    bool use_adj, int off_H0, int off_W0, int off_H1, int off_W1){
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(qinds);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  refinement_forward_bilin2d_cuda(vid0, vid1, qinds, dists, inds,
                                  ws_h, ws_w, ps, k, dist_type, stride0, stride1,
                                  dilation, pt, qshift, reflect_bounds, full_ws,
                                  use_adj, off_H0, off_W0, off_H1, off_W1);
}

void ref_bwd_dists(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor grad_dists,
    const torch::Tensor inds, int q_shift, int stride0,
    int nH0, int nW0, int ps, int pt, int dilation,
    bool reflect_bounds, bool use_adj, int off_H0, int off_W0,
    int off_H1, int off_W1, int dist_type, int imode) {
  CHECK_INPUT(grad_vid0);
  CHECK_INPUT(grad_vid1);
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(grad_dists);
  // CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  if(imode == 0){
    non_local_search_backward_cuda(
          grad_vid0, grad_vid1, vid0, vid1,
          grad_dists, inds, q_shift, stride0, nH0, nW0,
          ps, pt, dilation, reflect_bounds,
          use_adj, off_H0, off_W0,
          off_H1, off_W1, dist_type);
  }else if (imode == 1){
    ref_bwd_dists_bilin2d_cuda(
          grad_vid0, grad_vid1,
          vid0, vid1, grad_dists, inds,
          q_shift, stride0, nH0, nW0,
          ps, pt, dilation, reflect_bounds,
          use_adj, off_H0, off_W0,
          off_H1, off_W1, dist_type);
  }else if(imode == 2){
    // non_local_search_backward_bilin3d_cuda(
    //       grad_vid0, grad_vid1,
    //       grad_fflow, grad_bflow,
    //       vid0, vid1, fflow, bflow,
    //       grad_dists, grad_inds, inds,
    //       q_shift, stride0, nH0, nW0,
    //       ps, pt, dilation, reflect_bounds,
    //       use_adj, off_H0, off_W0,
    //       off_H1, off_W1, dist_type);
  }else{
    assert (1==0);
  }

}

void ref_bwd_inds(
    torch::Tensor grad_qinds, const torch::Tensor grad_inds,
    const torch::Tensor qinds, const torch::Tensor inds){
  CHECK_INPUT(grad_qinds);
  CHECK_INPUT(grad_inds);
  CHECK_INPUT(qinds);
  CHECK_INPUT(inds);
  ref_bwd_inds_cuda(grad_qinds, grad_inds, qinds, inds);
}


// python bindings
void init_refinement(py::module &m){
  m.def("refinement_forward", &refinement_forward,
        "Product Refine Forward (CUDA)");
  m.def("refinement_bilin2d_forward", &refinement_bilin2d_forward,
        "Product Refine Bilin2d Forward (CUDA)");
  m.def("ref_bwd_inds", &ref_bwd_inds,
        "Product Refine Backwards Indices (CUDA)");
  m.def("ref_bwd_dists", &ref_bwd_dists,
        "Product Refine Backwards Dists (CUDA)");
}
