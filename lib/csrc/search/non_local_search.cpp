
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

void non_local_search_int_forward_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor flows,
    torch::Tensor dists, torch::Tensor inds,
    int ps, int k, int stride0, int stride1, int dilation, int pt,
    bool reflect_bounds, bool full_ws, int patch_offset,
    int off_Hq, int off_Wq, int dist_type);

void non_local_search_bilin2d_forward_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor flows,
    torch::Tensor dists, torch::Tensor inds,
    int ps, int k, int stride0, float stride1,
    int dilation, int pt, bool reflect_bounds,
    bool full_ws, int patch_offset,
    int off_Hq, int off_Wq, int dist_type);

void non_local_search_int_vid_backward_cuda(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor grad_dists, const torch::Tensor inds,
    int ps, int pt, int stride0, int dilation,
    bool reflect_bounds, int patch_offset,
    int off_Hq, int off_Wq, int dist_type);

void non_local_search_bilin2d_vid_backward_cuda(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor grad_dists, const torch::Tensor inds,
    int wt, int ps, int pt, int stride0, int dilation,
    bool reflect_bounds, int patch_offset,
    int off_Hq, int off_Wq, int dist_type);

void non_local_search_bilin2d_vidflows_backward_cuda(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1, torch::Tensor grad_flows,
    const torch::Tensor vid0, const torch::Tensor vid1, const torch::Tensor flows,
    const torch::Tensor grad_dists, const torch::Tensor grad_inds,
    const torch::Tensor dists, const torch::Tensor inds,
    int wt, int ps, int pt, int stride0, int dilation,
    bool reflect_bounds, int patch_offset,
    int off_Hq, int off_Wq, int dist_type);


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void non_local_search_int_forward(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor flows, torch::Tensor dists, torch::Tensor inds,
    int ps, int k, int stride0, int stride1, int dilation, int pt,
    bool reflect_bounds, bool full_ws, int patch_offset,
    int off_Hq, int off_Wq, int dist_type){
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(flows);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  non_local_search_int_forward_cuda(vid0, vid1, flows, dists, inds,
                                    ps, k, stride0, stride1, dilation, pt,
                                    reflect_bounds, full_ws, patch_offset,
                                    off_Hq, off_Wq, dist_type);
}

void non_local_search_bilin2d_forward(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor flows, torch::Tensor dists, torch::Tensor inds,
    int ps, int k, int stride0, float stride1, int dilation, int pt,
    bool reflect_bounds, bool full_ws, int patch_offset,
    int off_Hq, int off_Wq, int dist_type){
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(flows);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  non_local_search_bilin2d_forward_cuda(vid0, vid1, flows, dists, inds,
                                        ps, k, stride0, stride1, dilation, pt,
                                        reflect_bounds, full_ws, patch_offset,
                                        off_Hq, off_Wq, dist_type);
}

void non_local_search_int_vid_backward(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor grad_dists, const torch::Tensor inds,
    int ps, int pt, int stride0, int dilation,
    bool reflect_bounds, int patch_offset,
    int off_Hq, int off_Wq, int dist_type) {

  CHECK_INPUT(grad_vid0);
  CHECK_INPUT(grad_vid1);
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(grad_dists);
  CHECK_INPUT(inds);
  non_local_search_int_vid_backward_cuda(grad_vid0, grad_vid1, vid0, vid1,
                                         grad_dists, inds, ps, pt, stride0, dilation,
                                         reflect_bounds, patch_offset,
                                         off_Hq, off_Wq, dist_type);

}

void non_local_search_bilin2d_vid_backward(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor grad_dists, const torch::Tensor inds,
    int wt, int ps, int pt, int stride0, int dilation,
    bool reflect_bounds, int patch_offset,
    int off_Hq, int off_Wq, int dist_type) {
  CHECK_INPUT(grad_vid0);
  CHECK_INPUT(grad_vid1);
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(grad_dists);
  CHECK_INPUT(inds);
  non_local_search_bilin2d_vid_backward_cuda(grad_vid0, grad_vid1,
                                             vid0, vid1, grad_dists, inds,
                                             wt, ps, pt, stride0, dilation,
                                             reflect_bounds, patch_offset,
                                             off_Hq, off_Wq, dist_type);
}

void non_local_search_bilin2d_vidflows_backward(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1, torch::Tensor grad_flows,
    const torch::Tensor vid0, const torch::Tensor vid1, const torch::Tensor flows,
    const torch::Tensor grad_dists, const torch::Tensor grad_inds,
    const torch::Tensor dists, const torch::Tensor inds,
    int wt, int ps, int pt, int stride0, int dilation,
    bool reflect_bounds, int patch_offset, int off_Hq, int off_Wq, int dist_type) {
  CHECK_INPUT(grad_vid0);
  CHECK_INPUT(grad_vid1);
  CHECK_INPUT(grad_flows);
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(flows);
  CHECK_INPUT(grad_dists);
  CHECK_INPUT(grad_inds);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  non_local_search_bilin2d_vidflows_backward_cuda(grad_vid0, grad_vid1, grad_flows,
                                                  vid0, vid1, flows,
                                                  grad_dists, grad_inds, dists, inds,
                                                  wt, ps, pt, stride0, dilation,
                                                  reflect_bounds, patch_offset,
                                                  off_Hq, off_Wq, dist_type);
}




// python bindings
void init_non_local_search(py::module &m){
  m.def("non_local_search_int_forward", &non_local_search_int_forward,
        "Search Forward with Heads (CUDA)");
  m.def("non_local_search_bilin2d_forward", &non_local_search_bilin2d_forward,
        "Search Forward with Heads (CUDA)");
  m.def("non_local_search_int_vid_backward",
        &non_local_search_int_vid_backward,
        "Search Backward (Vid0,Vid1)");
  m.def("non_local_search_bilin2d_vid_backward",
        &non_local_search_bilin2d_vid_backward,
        "Search Backward (Vid0,Vid1)");
  m.def("non_local_search_bilin2d_vidflows_backward",
        &non_local_search_bilin2d_vidflows_backward,
        "Search Backward (Vid0,Vid1,Flow)");
  // m.def("non_local_search_flow_backward", &non_local_search_flow_backward,
  //       "Search Backward (Flows)");

}

