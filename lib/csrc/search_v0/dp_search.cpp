#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void dp_search_forward_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor fflow, const torch::Tensor bflow,
    torch::Tensor dists, torch::Tensor inds,
    int wt, int ps, int k, int dist_type,
    int stride0, int stride1,
    int dilation, int pt, int qshift,
    bool reflect_bounds, bool ps_corner);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void dp_search_forward(
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor fflow, const torch::Tensor bflow,
    torch::Tensor dists, torch::Tensor inds,
    int wt, int ps, int k, int dist_type,
    int stride0, int stride1, int dilation, int pt, int qshift,
    bool reflect_bounds, bool ps_corner){
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(fflow);
  CHECK_INPUT(bflow);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  dp_search_forward_cuda(vid0, vid1, fflow, bflow, dists, inds,
                         wt, ps, k, dist_type,
                         stride0, stride1, dilation, pt, qshift,
                         reflect_bounds, ps_corner);
}


// python bindings
void init_dp_search(py::module &m){
  m.def("dp_search_forward", &non_local_search_forward,
        "Dynamic Programming Search (CUDA)");
}

