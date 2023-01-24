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

void optical_flow_accumulate_cuda(
    const torch::Tensor fflow,const torch::Tensor bflow,
    torch::Tensor pfflow, torch::Tensor pbflow,
    int stride0);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void optical_flow_accumulate(
    const torch::Tensor fflow, const torch::Tensor bflow,
    const torch::Tensor pfflow, const torch::Tensor pbflow,
    int stride0){
  CHECK_INPUT(fflow);
  CHECK_INPUT(bflow);
  CHECK_INPUT(pfflow);
  CHECK_INPUT(pbflow);
  optical_flow_accumulate_cuda(fflow, bflow, pfflow, pbflow, stride0);
}



// python bindings
void init_optical_flow_accumulate(py::module &m){
  m.def("optical_flow_accumulate", &optical_flow_accumulate,
        "Accumulate Offsets from Optical Flow (CUDA)");
}
