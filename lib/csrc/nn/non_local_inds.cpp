/*************************

Get indices of a non-local search

*****************************/

// -- included for pytorch --
#include <torch/extension.h>
#include <vector>

// -- include cuda_runtime for jax --
// #include <cuda_runtime_api.h>
// #include <cstddef>
// #include <cstdint>
// #include <cstdlib>
// #include <pybind11/pybind11.h>


// CUDA forward declarations

void non_local_inds_cuda(
    torch::Tensor inds, const torch::Tensor fflow, const torch::Tensor bflow,
    int ws, int wt, int stride0, float stride1, bool full_ws);

void non_local_flow_cuda(
    torch::Tensor inds, const torch::Tensor fflow, const torch::Tensor bflow,
    int ws, int wt, int stride0, float stride1, bool full_ws);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void non_local_inds(
    torch::Tensor inds,
    const torch::Tensor fflow, const torch::Tensor bflow,
    int ws, int wt, int stride0, float stride1, bool full_ws){
  CHECK_INPUT(inds);
  CHECK_INPUT(fflow);
  CHECK_INPUT(bflow);
  non_local_inds_cuda(inds, fflow, bflow,
                      ws, wt, stride0, stride1, full_ws);
}


void non_local_int_flow(
    torch::Tensor flow,
    const torch::Tensor fflow, const torch::Tensor bflow,
    int ws, int wt, int stride0, float stride1, bool full_ws){
  CHECK_INPUT(flow);
  CHECK_INPUT(fflow);
  CHECK_INPUT(bflow);
  non_local_flow_cuda(flow, fflow, bflow,
                      ws, wt, stride0, stride1, full_ws);
}


// python bindings
void init_non_local_inds(py::module &m){
  m.def("non_local_inds", &non_local_inds,
        "Get the Indices used for a Non-Local Search with Opitcal Flow (CUDA)");
  m.def("non_local_int_flow", &non_local_int_flow,
        "Get the Offsets used for a Non-Local Search with Opitcal Flow (CUDA)");
}
