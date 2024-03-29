// -- included for pytorch --
#include <torch/extension.h>
#include <vector>

// -- include cuda_runtime for jax --
// #include <cuda_runtime_api.h>
// #include <cstddef>
// #include <cstdint>
// #include <cstdlib>
// #include <pybind11/pybind11.h>
// using namespace torch::indexing;


// CUDA forward declarations

void accumulate_flow_forward_cuda(
    const torch::Tensor fflow,const torch::Tensor bflow,
    torch::Tensor pfflow, torch::Tensor pbflow,
    int stride0);

void accumulate_flow_backward_cuda(
    torch::Tensor dev,
    torch::Tensor grad_fflow, torch::Tensor grad_bflow,
    const torch::Tensor grad_pfflow, const torch::Tensor grad_pbflow,
    const torch::Tensor fflow, const torch::Tensor bflow,
    const torch::Tensor pfflow, const torch::Tensor pbflow,
    int stride0);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void accumulate_flow_forward(
    const torch::Tensor fflow, const torch::Tensor bflow,
    const torch::Tensor pfflow, const torch::Tensor pbflow,
    int stride0){
  CHECK_INPUT(fflow);
  CHECK_INPUT(bflow);
  CHECK_INPUT(pfflow);
  CHECK_INPUT(pbflow);
  accumulate_flow_forward_cuda(fflow, bflow, pfflow, pbflow, stride0);
}

void accumulate_flow_backward(
    torch::Tensor dev,
    torch::Tensor grad_fflow, torch::Tensor grad_bflow,
    const torch::Tensor grad_pfflow, const torch::Tensor grad_pbflow,
    const torch::Tensor fflow, const torch::Tensor bflow,
    const torch::Tensor pfflow, const torch::Tensor pbflow,
    int stride0){
  CHECK_INPUT(dev);
  CHECK_INPUT(grad_fflow);
  CHECK_INPUT(grad_bflow);
  CHECK_INPUT(grad_pfflow);
  CHECK_INPUT(grad_pbflow);
  CHECK_INPUT(fflow);
  CHECK_INPUT(bflow);
  CHECK_INPUT(pfflow);
  CHECK_INPUT(pbflow);
  accumulate_flow_backward_cuda(dev,grad_fflow,grad_bflow,
                                grad_pfflow,grad_pbflow,
                                fflow, bflow,
                                pfflow, pbflow, stride0);
}



// python bindings
void init_accumulate_flow(py::module &m){
  m.def("accumulate_flow_forward", &accumulate_flow_forward,
        "Accumulate Offsets from Optical Flow (CUDA)");
  m.def("accumulate_flow_backward", &accumulate_flow_backward,
        "Accumulate Offsets from Optical Flow (CUDA)");

}
