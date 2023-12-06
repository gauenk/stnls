#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

/*************************************

            Int Forward

 *************************************/

void gather_add_forward_cuda(
  torch::Tensor out_vid, torch::Tensor counts,
  const torch::Tensor in_vid,
  const torch::Tensor dists, const torch::Tensor inds,
  int ps, int strideIn, int strideOut, int pt, int dilation,
  bool reflect_bounds, int patch_offset, bool itype_int);

void gather_add_int_backward_cuda(
    torch::Tensor in_vid_grad,
    torch::Tensor dists_grad,
    const torch::Tensor out_vid_grad, const torch::Tensor vid,
    const torch::Tensor dists, const torch::Tensor inds,
    int ps, int strideIn, int strideOut, int pt, int dilation,
    bool reflect_bounds, int patch_offset);

void gather_add_bilin2d_backward_cuda(
    torch::Tensor in_vid_grad,
    torch::Tensor dists_grad,
    torch::Tensor inds_grad,
    const torch::Tensor out_vid_grad, const torch::Tensor vid,
    const torch::Tensor dists, const torch::Tensor inds,
    int ps, int strideIn, int strideOut, int pt, int dilation,
    bool reflect_bounds, int patch_offset);


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/***********************


      Int Indexing


***********************/


void gather_add_forward(
  torch::Tensor out_vid, torch::Tensor counts,
  const torch::Tensor in_vid, const torch::Tensor dists,
  const torch::Tensor inds, int ps, int strideIn, int strideOut, int pt,
  int dilation, bool reflect_bounds, int patch_offset, bool itype_int){
  CHECK_INPUT(out_vid);
  CHECK_INPUT(counts);
  CHECK_INPUT(in_vid);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  gather_add_int_forward_cuda(out_vid,counts,in_vid,dists,inds,
                              ps,strideIn,strideOut,pt,dilation,
                              reflect_bounds,patch_offset,itype_int);
}

void gather_add_int_backward(
  torch::Tensor in_vid_grad, torch::Tensor dists_grad,
  const torch::Tensor out_vid_grad, const torch::Tensor vid,
  const torch::Tensor dists, const torch::Tensor inds,
  int ps, int strideIn, int strideOut, int pt, int dilation,
  bool reflect_bounds, int patch_offset){
  CHECK_INPUT(in_vid_grad);
  CHECK_INPUT(dists_grad);
  CHECK_INPUT(out_vid_grad);
  CHECK_INPUT(vid);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  gather_add_int_backward_cuda(in_vid_grad,dists_grad,
                               out_vid_grad,vid,dists,inds,
                               ps,strideIn,strideOut,pt,dilation,
                               reflect_bounds,patch_offset);
}

void gather_add_bilin2d_backward( // "in" and "out" w.r.t. forward pass
  torch::Tensor in_vid_grad,
  torch::Tensor dists_grad, torch::Tensor inds_grad,
  const torch::Tensor out_vid_grad, const torch::Tensor vid,
  const torch::Tensor dists, const torch::Tensor inds,
  int ps, int strideIn, int strideOut, int pt, int dilation,
  bool reflect_bounds, int patch_offset){
  CHECK_INPUT(in_vid_grad);
  CHECK_INPUT(dists_grad);
  CHECK_INPUT(inds_grad);
  CHECK_INPUT(out_vid_grad);
  CHECK_INPUT(vid);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  gather_add_bilin2d_backward_cuda(in_vid_grad,dists_grad,inds_grad,
                                   out_vid_grad,vid,dists,inds,
                                   ps,strideIn,strideOut,pt,dilation,
                                   reflect_bounds,patch_offset);
}

/***********************


    Python Bindings


***********************/

void init_gather_add(py::module &m){
  m.def("gather_add_forward", &gather_add_forward,
        "WeightedPatchSum Forward (CUDA)");
  m.def("gather_add_int_backward", &gather_add_int_backward,
        "WeightedPatchSum Backward (CUDA)");
  m.def("gather_add_bilin2d_backward", &gather_add_bilin2d_backward,
        "WeightedPatchSum Backward (CUDA)");

}

