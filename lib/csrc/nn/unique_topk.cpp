#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void unique_topk_forward_cuda(
     const torch::Tensor dists, const torch::Tensor inds,
     torch::Tensor dists_topk, torch::Tensor inds_topk,
     int k);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void unique_topk_forward(const torch::Tensor dists,
                         const torch::Tensor inds,
                         torch::Tensor dists_topk,
                         torch::Tensor inds_topk,
                         int k){
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  CHECK_INPUT(dists_topk);
  CHECK_INPUT(inds_topk);
  unique_topk_forward_cuda(dists,inds,dists_topk,inds_topk,k);
}

// python bindings
void init_unique_topk(py::module &m){
  m.def("unique_topk", &unique_topk_forward,
        "Unique Top-K");
}
