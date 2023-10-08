#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void paired_search_int_forward_cuda(
    const torch::Tensor frame0, const torch::Tensor frame1,
    const torch::Tensor flow, torch::Tensor dists, torch::Tensor inds,
    int ps, int k, int stride0, int stride1, int dilation,
    bool reflect_bounds, bool full_ws, int patch_offset, int dist_type);

void paired_search_bilin2d_forward_cuda(
    const torch::Tensor frame0, const torch::Tensor frame1,
    const torch::Tensor flow, torch::Tensor dists, torch::Tensor inds,
    int ps, int k, int stride0, float stride1, int dilation,
    bool reflect_bounds, bool full_ws, int patch_offset, int dist_type);

void paired_search_backward_cuda(
    torch::Tensor grad_frame0, torch::Tensor grad_frame1,
    const torch::Tensor frame0, const torch::Tensor frame1,
    const torch::Tensor grad_dists, const torch::Tensor inds,
    int stride0, int ps, int dilation, bool reflect_bounds,
    int patch_size, int dist_type);

void paired_search_bilin2d_backward_cuda(
    torch::Tensor grad_frame0, torch::Tensor grad_frame1,
    torch::Tensor grad_flow,
    const torch::Tensor frame0, const torch::Tensor frame1,
    const torch::Tensor flow,
    const torch::Tensor grad_dists, const torch::Tensor grad_inds,
    const torch::Tensor inds,
    int stride0, int ps, int dilation, bool reflect_bounds,
    int patch_size, int dist_type);


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void paired_search_int_forward(
    const torch::Tensor frame0, const torch::Tensor frame1,
    const torch::Tensor flow, torch::Tensor dists, torch::Tensor inds,
    int ps, int k, int stride0, int stride1, int dilation,
    bool reflect_bounds, bool full_ws, int patch_offset, int dist_type){
  CHECK_INPUT(frame0);
  CHECK_INPUT(frame1);
  CHECK_INPUT(flow);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  paired_search_int_forward_cuda(frame0, frame1, flow, dists, inds,
                                 ps, k, stride0, stride1, dilation,
                                 reflect_bounds, full_ws, patch_offset, dist_type);

}

void paired_search_bilin2d_forward(
    const torch::Tensor frame0, const torch::Tensor frame1,
    const torch::Tensor flow, torch::Tensor dists, torch::Tensor inds,
    int ps, int k, int stride0, float stride1, int dilation,
    bool reflect_bounds, bool full_ws, int patch_offset, int dist_type){
  CHECK_INPUT(frame0);
  CHECK_INPUT(frame1);
  CHECK_INPUT(flow);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  paired_search_bilin2d_forward_cuda(frame0, frame1, flow, dists, inds,
                                     ps, k, stride0, stride1, dilation,
                                     reflect_bounds, full_ws, patch_offset, dist_type);
}

void paired_search_backward(
    torch::Tensor grad_frame0,
    torch::Tensor grad_frame1,
    torch::Tensor grad_flow,
    const torch::Tensor frame0, const torch::Tensor frame1,
    const torch::Tensor flow,
    const torch::Tensor grad_dists, const torch::Tensor grad_inds,
    const torch::Tensor inds,
    int stride0, int ps, int dilation, bool reflect_bounds,
    int patch_size, int dist_type) {

  CHECK_INPUT(grad_frame0);
  CHECK_INPUT(grad_frame1);
  CHECK_INPUT(grad_flow);
  CHECK_INPUT(frame0);
  CHECK_INPUT(frame1);
  CHECK_INPUT(flow);
  CHECK_INPUT(grad_dists);
  CHECK_INPUT(grad_inds);
  CHECK_INPUT(inds);

  if(inds.dtype() == torch::kInt32){
    assert(1==0);
    // paired_search_backward_cuda(
    //       grad_frame0, grad_frame1, frame0, frame1,
    //       grad_dists, inds, q_shift, stride0, nH0, nW0,
    //       ps, dilation, reflect_bounds,
    //       use_adj, off_H0, off_W0,
    //       off_H1, off_W1, dist_type);
  }else{
    paired_search_bilin2d_backward_cuda(
           grad_frame0, grad_frame1, grad_flow,
           frame0, frame1, flow,
           grad_dists, grad_inds, inds,
           stride0, ps, dilation, reflect_bounds,
           patch_size, dist_type);
  }

}


// python bindings
void init_paired_search(py::module &m){
  m.def("paired_search_int_forward", &paired_search_int_forward,
        "Search Forward with Heads (CUDA)");
  m.def("paired_search_bilin2d_forward", &paired_search_bilin2d_forward,
        "Search Forward with Heads (CUDA)");
  m.def("paired_search_backward", &paired_search_backward,
        "Search Backward with Heads (CUDA)");
}

