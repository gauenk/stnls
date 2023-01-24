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

void temporal_inds_cuda(
    const torch::Tensor inds,
    const torch::Tensor fflow,
    const torch::Tensor bflow,
    torch::Tensor inds_t);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void temporal_inds(
    const torch::Tensor inds,
    const torch::Tensor fflow,
    const torch::Tensor bflow,
    torch::Tensor inds_t){
  CHECK_INPUT(inds);
  CHECK_INPUT(fflow);
  CHECK_INPUT(bflow);
  CHECK_INPUT(inds_t);
  temporal_inds_cuda(inds, fflow, bflow, inds_t);
}



// python bindings
void init_temporal_inds(py::module &m){
  m.def("temporal_inds", &temporal_inds,
        "Get the Temporal Index Offsets using Opitcal Flow (CUDA)");
}
