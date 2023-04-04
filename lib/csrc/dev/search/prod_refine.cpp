#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void prod_refine_forward_cuda(
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor dists, torch::Tensor inds,
    torch::Tensor self_dists, torch::Tensor qinds,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int ws_h, int ws_w,
    int ws_h_og, int ws_w_og, int chnls,
    int dilation, int stride1, bool use_adj,
    bool reflect_bounds, bool search_abs,
    bool full_ws, bool anchor_self, bool use_self);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

        // stnls_cuda.prod_refine_forward(vid0, vid1, dists_exh, inds_exh,
        //                               self_dists, qinds,
        //                               qstart, stride0, n_h0, n_w0,
        //                               h0_off, w0_off, h1_off, w1_off,
        //                               ps, pt, ws_h, ws_w,
        //                               chnls, dilation, stride1, use_adj,
        //                               reflect_bounds, search_abs, full_ws,
        //                               anchor_self, use_self)

void prod_refine_forward(
    torch::Tensor vid0, torch::Tensor vid1,
    torch::Tensor dists, torch::Tensor inds,
    torch::Tensor self_dists, torch::Tensor qinds,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int ws_h, int ws_w,
    int ws_h_og, int ws_w_og, int chnls, int dilation, int stride1,
    bool use_adj, bool reflect_bounds, bool search_abs,
    bool full_ws, bool anchor_self, bool use_self){
  CHECK_INPUT(vid0);
  CHECK_INPUT(vid1);
  CHECK_INPUT(dists);
  CHECK_INPUT(inds);
  CHECK_INPUT(self_dists);
  CHECK_INPUT(qinds);
  prod_refine_forward_cuda(vid0,vid1,dists,inds,self_dists, qinds,
                           qstart, stride0, n_h0, n_w0,
                           h0_off,w0_off,h1_off,w1_off,
                           ps,pt,ws_h,ws_w,ws_h_og,ws_w_og,
                           chnls,dilation,stride1,use_adj,reflect_bounds,search_abs,
                           full_ws,anchor_self,use_self);
}

// python bindings
void init_prod_refine(py::module &m){
  m.def("prod_refine_forward", &prod_refine_forward,
        "Product Refine Forward (CUDA)");
}
