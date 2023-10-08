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

void search_flow_forward_cuda(
    const torch::Tensor fflow, const torch::Tensor bflow,
    torch::Tensor flows, int wt, int stride0);

void search_flow_backward_cuda(
    torch::Tensor grad_fflow, torch::Tensor grad_bflow,
    const torch::Tensor grad_flows,
    const torch::Tensor fflow, const torch::Tensor bflow,
    const torch::Tensor flows,
    int wt, int stride0);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void search_flow_forward(
    const torch::Tensor fflow, const torch::Tensor bflow,
    torch::Tensor flows, int wt, int stride0){
  CHECK_INPUT(fflow);
  CHECK_INPUT(bflow);
  CHECK_INPUT(flows);
  search_flow_forward_cuda(fflow, bflow, flows, wt, stride0);
}

void search_flow_backward(
    torch::Tensor grad_fflow, torch::Tensor grad_bflow,
    const torch::Tensor grad_flows,
    const torch::Tensor fflow, const torch::Tensor bflow,
    const torch::Tensor flows,
    int wt, int stride0){
  CHECK_INPUT(grad_fflow);
  CHECK_INPUT(grad_bflow);
  CHECK_INPUT(grad_flows);
  CHECK_INPUT(fflow);
  CHECK_INPUT(bflow);
  CHECK_INPUT(flows);
  search_flow_backward_cuda(grad_fflow,grad_bflow,
                            grad_flows, fflow, bflow,
                            flows, wt, stride0);
}



// python bindings
void init_search_flow(py::module &m){
  m.def("search_flow_forward", &search_flow_forward,
        "Accumulate Offsets from Optical Flow (CUDA)");
  m.def("search_flow_backward", &search_flow_backward,
        "Accumulate Offsets from Optical Flow (CUDA)");

}
