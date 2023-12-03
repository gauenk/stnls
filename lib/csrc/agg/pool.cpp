#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

/*************************************

            Int Forward

 *************************************/

void pool_int_forward_cuda(
  torch::Tensor out_vid, torch::Tensor counts,
  const torch::Tensor in_vid,
  const torch::Tensor dists, const torch::Tensor inds,
  int ps, int stride0, int pt, int dilation,
  bool reflect_bounds, int patch_offset);

void pool_int_backward_cuda(
    torch::Tensor in_vid_grad,
    torch::Tensor dists_grad,
    const torch::Tensor out_vid_grad, const torch::Tensor vid,
    const torch::Tensor dists, const torch::Tensor inds,
    int ps, int stride0, int pt, int dilation, bool reflect_bounds, int patch_offset);

/*************************************

           Bilin2d Forward

 *************************************/

// void pool_bilin2d_forward_cuda(
//   torch::Tensor out_vid, torch::Tensor counts,
//   const torch::Tensor in_vid,
//   const torch::Tensor dists, const torch::Tensor inds,
//   int ps, int stride0, int pt, int dilation,
//   bool reflect_bounds, int patch_offset);

// void pool_bilin2d_backward_cuda(
//     torch::Tensor in_vid_grad,
//     torch::Tensor dists_grad,
//     torch::Tensor inds_grad,
//     const torch::Tensor out_vid_grad, const torch::Tensor vid,
//     const torch::Tensor dists, const torch::Tensor inds,
//     int ps, int stride0, int pt, int dilation, bool reflect_bounds, int patch_offset);


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/***********************


      Int Indexing


***********************/


void pool_int_forward(
  torch::Tensor out_vid, torch::Tensor counts,
  const torch::Tensor in_vid,
  const torch::Tensor dists,
  const torch::Tensor inds,
  int ps, int stride0, int pt, int dilation,
  bool reflect_bounds, int patch_offset){
  CHECK_INPUT(out_vid);
  CHECK_INPUT(counts);
  CHECK_INPUT(in_vid);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  pool_int_forward_cuda(out_vid,counts,in_vid,dists,inds,
                         ps,stride0,pt,dilation,reflect_bounds,patch_offset);
}

void pool_int_backward( // "in" and "out" w.r.t. forward pass
  torch::Tensor in_vid_grad, torch::Tensor dists_grad,
  const torch::Tensor out_vid_grad, const torch::Tensor vid,
  const torch::Tensor dists, const torch::Tensor inds,
  int ps, int stride0, int pt, int dilation, bool reflect_bounds, int patch_offset){
  CHECK_INPUT(in_vid_grad);
  CHECK_INPUT(dists_grad);
  CHECK_INPUT(out_vid_grad);
  CHECK_INPUT(vid);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  pool_int_backward_cuda(in_vid_grad,dists_grad,
                          out_vid_grad,vid,dists,inds,
                          ps,stride0,pt,dilation,reflect_bounds,patch_offset);
}

/***********************


      Bilinear2d


***********************/

void pool_bilin2d_forward(
  torch::Tensor out_vid, torch::Tensor counts,
  const torch::Tensor in_vid,
  const torch::Tensor dists,
  const torch::Tensor inds,
  int ps, int stride0, int pt, int dilation,
  bool reflect_bounds, int patch_offset){
  CHECK_INPUT(out_vid);
  CHECK_INPUT(counts);
  CHECK_INPUT(in_vid);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  // pool_bilin2d_forward_cuda(out_vid,counts,in_vid,dists,inds,
  //                        ps,stride0,pt,dilation,reflect_bounds,patch_offset);
}

void pool_bilin2d_backward( // "in" and "out" w.r.t. forward pass
  torch::Tensor in_vid_grad,
  torch::Tensor dists_grad, torch::Tensor inds_grad,
  const torch::Tensor out_vid_grad, const torch::Tensor vid,
  const torch::Tensor dists, const torch::Tensor inds,
  int ps, int stride0, int pt, int dilation, bool reflect_bounds, int patch_offset){
  CHECK_INPUT(in_vid_grad);
  CHECK_INPUT(dists_grad);
  CHECK_INPUT(inds_grad);
  CHECK_INPUT(out_vid_grad);
  CHECK_INPUT(vid);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  // pool_bilin2d_backward_cuda(in_vid_grad,dists_grad,
  //                             inds_grad,
  //                             out_vid_grad,vid,dists,inds,
  //                             ps,stride0,pt,dilation,reflect_bounds,patch_offset);
}

/***********************


    Python Bindings


***********************/

void init_pool(py::module &m){
  m.def("pool_int_forward", &pool_int_forward,
        "WeightedPatchSum Forward (CUDA)");
  m.def("pool_int_backward", &pool_int_backward,
        "WeightedPatchSum Backward (CUDA)");
  // m.def("pool_bilin2d_forward", &pool_bilin2d_forward,
  //       "WeightedPatchSum Forward (CUDA)");
  // m.def("pool_bilin2d_backward", &pool_bilin2d_backward,
  //       "WeightedPatchSum Backward (CUDA)");

}

