// -- included for pytorch --
#include <torch/extension.h>
#include <vector>

// -- include cuda_runtime for jax --
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <pybind11/pybind11.h>
#include "../jax_pybind.h"
using namespace torch::indexing;


// CUDA forward declarations

void jitter_unique_inds_cuda(
    torch::Tensor inds, int K,
    int H, int W);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void jitter_unique_inds(
    torch::Tensor inds, int K,
    int H, int W){
  CHECK_INPUT(inds);
  jitter_unique_inds_cuda(inds, K, H, W);
}



// python bindings
void init_jitter_unique_inds(py::module &m){
  m.def("jitter_unique_inds", &jitter_unique_inds,
        "Get unique indices by jitter existing ones. (CUDA)");
}
