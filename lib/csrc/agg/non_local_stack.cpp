// imports
#include <torch/extension.h>
// #include <torch/types.h>
#include <vector>
// #include "pybind.hpp"


// CUDA forward declarations

void non_local_stack_int_forward_cuda(
    const torch::Tensor vid, const torch::Tensor weights,
    const torch::Tensor inds, torch::Tensor stack, torch::Tensor counts,
    int ps, int pt, int dilation, int stride0,
    bool reflect_bounds, int patch_offset);

void non_local_stack_bilin2d_forward_cuda(
    const torch::Tensor vid, const torch::Tensor weights,
    const torch::Tensor inds, torch::Tensor stack, torch::Tensor counts,
    int ps, int pt, int dilation, int stride0,
    bool reflect_bounds, int patch_offset);

void non_local_stack_int_backward_cuda(
    torch::Tensor grad_vid,
    torch::Tensor grad_weights,
    const torch::Tensor grad_stack,
    const torch::Tensor vid,
    const torch::Tensor weights,
    const torch::Tensor inds,
    const torch::Tensor stack,
    const torch::Tensor counts,
    int ps, int pt, int dilation, int stride0,
    bool reflect_bounds, int patch_offset);

void non_local_stack_bilin2d_backward_cuda(
    torch::Tensor grad_vid,
    torch::Tensor grad_weights,
    torch::Tensor grad_inds,
    const torch::Tensor grad_stack,
    const torch::Tensor vid,
    const torch::Tensor weights,
    const torch::Tensor inds,
    const torch::Tensor stack,
    const torch::Tensor counts,
    int ps, int pt, int dilation, int stride0,
    bool reflect_bounds, int patch_offset);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/*********************************

      Using Raster Order

*********************************/

void non_local_stack_int_forward(
    const torch::Tensor vid, const torch::Tensor weights,
    const torch::Tensor inds, torch::Tensor stack, torch::Tensor counts,
    int ps, int pt, int dilation, int stride0, bool reflect_bounds, int patch_offset){
  CHECK_INPUT(vid);
  CHECK_INPUT(weights);
  CHECK_INPUT(inds);
  CHECK_INPUT(stack);
  CHECK_INPUT(counts);

    non_local_stack_int_forward_cuda(vid,weights,inds,stack,counts,
                                     ps,pt,dilation,stride0,
                                     reflect_bounds,patch_offset);
}


void non_local_stack_bilin2d_forward(
    const torch::Tensor vid, const torch::Tensor weights,
    const torch::Tensor inds, torch::Tensor stack, torch::Tensor counts,
    int ps, int pt, int dilation, int stride0, bool reflect_bounds, int patch_offset){
  CHECK_INPUT(vid);
  CHECK_INPUT(weights);
  CHECK_INPUT(inds);
  CHECK_INPUT(stack);
  CHECK_INPUT(counts);

    non_local_stack_bilin2d_forward_cuda(vid,weights,inds,stack,counts,
                                         ps,pt,dilation,stride0,
                                         reflect_bounds,patch_offset);
}


void non_local_stack_int_backward(
    torch::Tensor grad_vid,
    torch::Tensor grad_weights,
    const torch::Tensor grad_stack,
    const torch::Tensor vid,
    const torch::Tensor weights,
    const torch::Tensor inds,
    const torch::Tensor stack,
    const torch::Tensor counts,
    int ps, int pt, int dilation, int stride0,
    bool reflect_bounds, int patch_offset){
  CHECK_INPUT(grad_vid);
  CHECK_INPUT(grad_weights);
  CHECK_INPUT(grad_stack);
  CHECK_INPUT(vid);
  CHECK_INPUT(weights);
  CHECK_INPUT(inds);
  CHECK_INPUT(stack);
  CHECK_INPUT(counts);
  non_local_stack_int_backward_cuda(grad_vid,grad_weights,grad_stack,
                                    vid,weights,inds,stack,counts,
                                    ps,pt,dilation,stride0,
                                    reflect_bounds,patch_offset);
}


void non_local_stack_bilin2d_backward(
    torch::Tensor grad_vid,
    torch::Tensor grad_weights,
    torch::Tensor grad_inds,
    const torch::Tensor grad_stack,
    const torch::Tensor vid,
    const torch::Tensor weights,
    const torch::Tensor inds,
    const torch::Tensor stack,
    const torch::Tensor counts,
    int ps, int pt, int dilation, int stride0,
    bool reflect_bounds, int patch_offset){
  CHECK_INPUT(grad_vid);
  CHECK_INPUT(grad_weights);
  CHECK_INPUT(grad_inds);
  CHECK_INPUT(grad_stack);
  CHECK_INPUT(vid);
  CHECK_INPUT(weights);
  CHECK_INPUT(inds);
  CHECK_INPUT(stack);
  CHECK_INPUT(counts);
  non_local_stack_bilin2d_backward_cuda(grad_vid,grad_weights,grad_inds,grad_stack,
                                        vid,weights,inds,stack,counts,
                                        ps,pt,dilation,stride0,
                                        reflect_bounds,patch_offset);
}


// python bindings
void init_non_local_stack(py::module &m){
  m.def("non_local_stack_int_forward", &non_local_stack_int_forward,
        "NLS Stack Forward with Int Indexing");
  m.def("non_local_stack_bilin2d_forward", &non_local_stack_bilin2d_forward,
        "NLS Stack Forward with Bilin2d Indexing");
  m.def("non_local_stack_int_backward",&non_local_stack_int_backward,
        "NLS Stack Backward with Int Indexing");
  m.def("non_local_stack_bilin2d_backward",&non_local_stack_bilin2d_backward,
        "NLS Stack Backward with Bilin2d Indexing");
}

