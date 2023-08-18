// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
// #include "shared_kernel.cu"
#include "nls_bilin2d.cu"
using namespace at;

/************************************

   Backward Dists (bilinear 2d)

************************************/


template <typename scalar_t, int DIST_TYPE>
__global__ void ref_bwd_dists_bilin2d_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid0,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid1,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_dists,
    // const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_inds,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> inds,
    // torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> count0,
    // torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> count1,
    int q_shift, int stride0, int nH0, int nW0, int nHW0,
    int off_H0, int off_W0, int off_H1, int off_W1,
    int ps, int pt, int dilation, int patch_offset,
    bool reflect_bounds, int ftrs_per_thread) {

  // -- shape --
  int nbatch = grad_dists.size(0);
  int Q = grad_dists.size(2);
  int K =  grad_dists.size(3);
  int HD = vid0.size(1);
  int T = vid0.size(2);
  int F = vid0.size(3);
  int H = vid0.size(4);
  int W = vid0.size(5);

  // -- fwd decl registers --
  int ref_patch[3];
  scalar_t prop_patch[3];
  int ref[3];
  scalar_t prop[3];
  int prop_i[3];
  bool valid_ref[4];
  bool valid_prop[4];
  int qindex,qindex_tmp;

  bool valid;
  scalar_t weight,pix0,pix1,pix;
  // scalar_t iweight[3];
  int iftr;
  int center_offsets[4] = {off_H0,off_H1,off_W0,off_W1};

  // -- location to fill --
  int i0 = blockIdx.x*blockDim.x+threadIdx.x;
  int i1 = blockIdx.y*blockDim.y+threadIdx.y;
  int ihead = blockIdx.z/nbatch;
  int ibatch = (blockIdx.z-ihead*nbatch) % nbatch;

  // -- feature chunk --
  int ftr_start = threadIdx.z * ftrs_per_thread;
  int ftr_end = min(F,ftr_start + ftrs_per_thread);

  // -- each region --
  if ((i0 < Q) && (i1 < K)){

    // -- full-resolution video query index --
    qindex = i0 + q_shift;

    // -- pixel location from query index --
    get_pixel_loc(ref_patch,qindex,qindex_tmp,stride0,nW0,nHW0,H,W);

    // -- read from tensors --
    weight = grad_dists[ibatch][ihead][i0][i1];
  #pragma unroll
    for (int _idx=0; _idx < 3; _idx++){
      prop_patch[_idx] = inds[ibatch][ihead][i0][i1][_idx];
      // iweight[_idx] = grad_inds[ibatch][ihead][i0][i1][_idx];
    }

    // -- update vid0,vid1 --
    update_bwd_patch_bilin2d<scalar_t,DIST_TYPE>(
                     grad_vid0[ibatch][ihead],grad_vid1[ibatch][ihead],
                     vid0[ibatch][ihead],vid1[ibatch][ihead],
                     // count0[ibatch][ihead],count1[ibatch][ihead],
                     weight,ref_patch,prop_patch,
                     ps,pt,dilation,reflect_bounds,
                     center_offsets,patch_offset,
                     iftr,ftr_start,ftr_end,
                     ref,prop,prop_i,
                     valid_ref,valid_prop,valid,
                     T,H,W,pix0,pix1,pix,i1);


  }
}

void ref_bwd_dists_bilin2d_cuda(
    torch::Tensor grad_vid0, torch::Tensor grad_vid1,
    const torch::Tensor vid0, const torch::Tensor vid1,
    const torch::Tensor grad_dists, const torch::Tensor inds,
    int q_shift, int stride0, int nH0, int nW0,
    int ps, int pt, int dilation, bool reflect_bounds, bool use_adj,
    int off_H0, int off_W0, int off_H1, int off_W1, int dist_type) {

  // -- unpack --
  int B = vid0.size(0);
  int HD = vid0.size(1);
  int T = vid0.size(2);
  int F = vid0.size(3);
  int H = vid0.size(4);
  int W = vid0.size(5);
  int nqueries = inds.size(2);
  int K = inds.size(3);
  int BHD = B*HD;
  int nHW0 = nH0 * nW0;
  assert(pt == 1);

  // -- launch parameters --
  int nbatch = grad_dists.size(0);
  int nheads = grad_dists.size(1);
  int nq = grad_dists.size(2);
  int k = grad_dists.size(3);
  int ftr_threads = min(15,F);
  dim3 threadsPerBlock(10,4,ftr_threads);
  dim3 blocksPerGrid(1, 1, nheads*nbatch);
  blocksPerGrid.x = ceil(double(nq)/double(threadsPerBlock.x));
  blocksPerGrid.y = ceil(double(k)/double(threadsPerBlock.y));
  int ftrs_per_thread = (F-1)/ftr_threads+1;

  // -- shared --
  int psHalf = ps/2;
  int adj = use_adj ? psHalf : 0;
  int patch_offset = adj - psHalf;

  // -- launch kernel --
  if (dist_type == 0){ // prod
    AT_DISPATCH_FLOATING_TYPES(vid0.type(),
                               "ref_bwd_dists_bilin2d_kernel", ([&] {
    ref_bwd_dists_bilin2d_kernel<scalar_t,0>
      <<<blocksPerGrid, threadsPerBlock>>>(
          grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          // count0.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          // count1.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          q_shift, stride0, nH0, nW0, nHW0, off_H0, off_W0, off_H1, off_W1,
          ps, pt, dilation, patch_offset, reflect_bounds, ftrs_per_thread);
    }));
  }else if (dist_type == 1){ // l2
    AT_DISPATCH_FLOATING_TYPES(vid0.type(),
                               "ref_bwd_dists_bilin2d_kernel", ([&] {
    ref_bwd_dists_bilin2d_kernel<scalar_t,1>
      <<<blocksPerGrid, threadsPerBlock>>>(
          grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          // count0.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          // count1.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          q_shift, stride0, nH0, nW0, nHW0, off_H0, off_W0, off_H1, off_W1,
          ps, pt, dilation, patch_offset, reflect_bounds, ftrs_per_thread);
    }));
  }else{
     throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");    }


}

/************************************

   Backward Dists (bilinear 3d)

************************************/


// template <typename scalar_t, int DIST_TYPE>
// __global__ void ref_bwd_dists_bilin3d_kernel(
//     torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid0,
//     torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid1,
//     const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
//     const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
//     const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_dists,
//     const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_inds,
//     const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> inds,
//     // torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> count0,
//     // torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> count1,
//     int q_shift, int stride0, int nH0, int nW0, int nHW0,
//     int off_H0, int off_W0, int off_H1, int off_W1,
//     int ps, int pt, int dilation, int patch_offset,
//     bool reflect_bounds, int ftrs_per_thread) {

//   // -- shape --
//   int nbatch = grad_dists.size(0);
//   int Q = grad_dists.size(2);
//   int K =  grad_dists.size(3);
//   int HD = vid0.size(1);
//   int T = vid0.size(2);
//   int F = vid0.size(3);
//   int H = vid0.size(4);
//   int W = vid0.size(5);

//   // -- fwd decl registers --
//   int ref_patch[3];
//   scalar_t prop_patch[3];
//   int ref[3];
//   scalar_t prop[3];
//   int prop_i[3];
//   bool valid_ref[4];
//   bool valid_prop[4];
//   int qindex,qindex_tmp;

//   bool valid;
//   scalar_t weight,pix0,pix1,pix;
//   scalar_t iweight[3];
//   int iftr;
//   int center_offsets[4] = {off_H0,off_H1,off_W0,off_W1};

//   // -- location to fill --
//   int i0 = blockIdx.x*blockDim.x+threadIdx.x;
//   int i1 = blockIdx.y*blockDim.y+threadIdx.y;
//   int ihead = blockIdx.z/nbatch;
//   int ibatch = (blockIdx.z-ihead*nbatch) % nbatch;

//   // -- feature chunk --
//   int ftr_start = threadIdx.z * ftrs_per_thread;
//   int ftr_end = min(F,ftr_start + ftrs_per_thread);

//   // -- each region --
//   if ((i0 < Q) && (i1 < K)){

//     // -- full-resolution video query index --
//     qindex = i0 + q_shift;

//     // -- pixel location from query index --
//     get_pixel_loc(ref_patch,qindex,qindex_tmp,stride0,nW0,nHW0,H,W);

//     // -- read from tensors --
//     weight = grad_dists[ibatch][ihead][i0][i1];
//   #pragma unroll
//     for (int _idx=0; _idx < 3; _idx++){
//       prop_patch[_idx] = inds[ibatch][ihead][i0][i1][_idx];
//       iweight[_idx] = grad_inds[ibatch][ihead][i0][i1][_idx];
//     }

//     // -- update vid0,vid1 --
//     update_bwd_patch_bilin3d<scalar_t,DIST_TYPE>(
//                      grad_vid0[ibatch][ihead],grad_vid1[ibatch][ihead],
//                      vid0[ibatch][ihead],vid1[ibatch][ihead],
//                      // count0[ibatch][ihead],count1[ibatch][ihead],
//                      weight,ref_patch,prop_patch,
//                      ps,pt,dilation,reflect_bounds,
//                      center_offsets,patch_offset,
//                      iftr,ftr_start,ftr_end,
//                      ref,prop,prop_i,
//                      valid_ref,valid_prop,valid,
//                      T,H,W,pix0,pix1,pix,i1);


//   }
// }

// void ref_bwd_dists_bilin3d_cuda(
//     torch::Tensor grad_vid0, torch::Tensor grad_vid1,
//     const torch::Tensor vid0, const torch::Tensor vid1,
//     const torch::Tensor grad_dists, const torch::Tensor grad_inds,
//     const torch::Tensor inds, int q_shift, int stride0, int nH0, int nW0,
//     int ps, int pt, int dilation, bool reflect_bounds, bool use_adj,
//     int off_H0, int off_W0, int off_H1, int off_W1, int dist_type) {

//   // -- unpack --
//   int B = vid0.size(0);
//   int HD = vid0.size(1);
//   int T = vid0.size(2);
//   int F = vid0.size(3);
//   int H = vid0.size(4);
//   int W = vid0.size(5);
//   int nqueries = inds.size(2);
//   int K = inds.size(3);
//   int BHD = B*HD;
//   int nHW0 = nH0 * nW0;
//   assert(pt == 1);

//   // -- launch parameters --
//   int nbatch = grad_dists.size(0);
//   int nheads = grad_dists.size(1);
//   int nq = grad_dists.size(2);
//   int k = grad_dists.size(3);
//   int ftr_threads = min(15,F);
//   dim3 threadsPerBlock(10,4,ftr_threads);
//   dim3 blocksPerGrid(1, 1, nheads*nbatch);
//   blocksPerGrid.x = ceil(double(nq)/double(threadsPerBlock.x));
//   blocksPerGrid.y = ceil(double(k)/double(threadsPerBlock.y));
//   int ftrs_per_thread = (F-1)/ftr_threads+1;

//   // -- shared --
//   int psHalf = ps/2;
//   int adj = use_adj ? psHalf : 0;
//   int patch_offset = adj - psHalf;

//   // -- launch kernel --
//   if (dist_type == 0){ // prod
//     AT_DISPATCH_FLOATING_TYPES(vid0.type(),
//                                "ref_bwd_dists_bilin3d_kernel", ([&] {
//     ref_bwd_dists_bilin3d_kernel<scalar_t,0>
//       <<<blocksPerGrid, threadsPerBlock>>>(
//           grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
//           grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
//           vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
//           vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
//           grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//           grad_inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
//           inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
//           // count0.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
//           // count1.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
//           q_shift, stride0, nH0, nW0, nHW0, off_H0, off_W0, off_H1, off_W1,
//           ps, pt, dilation, patch_offset, reflect_bounds, ftrs_per_thread);
//     }));
//   }else if (dist_type == 1){ // l2
//     AT_DISPATCH_FLOATING_TYPES(vid0.type(),
//                                "ref_bwd_dists_bilin3d_kernel", ([&] {
//     ref_bwd_dists_bilin3d_kernel<scalar_t,1>
//       <<<blocksPerGrid, threadsPerBlock>>>(
//           grad_vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
//           grad_vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
//           vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
//           vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
//           grad_dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//           grad_inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
//           inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
//           // count0.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
//           // count1.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
//           q_shift, stride0, nH0, nW0, nHW0, off_H0, off_W0, off_H1, off_W1,
//           ps, pt, dilation, patch_offset, reflect_bounds, ftrs_per_thread);
//     }));
//   }else{
//      throw std::invalid_argument("Uknown distance type. Must be 0 (product) or 1 (l2)");    }


// }


/****************************

       Backward Indices 

****************************/

template <typename scalar_t>
__global__ void ref_bwd_inds_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_qinds,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_inds,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> qinds,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> inds,
    int num_per_thread){

  // -- unpack shape --
  int Ksearch = qinds.size(3);
  int Kagg = inds.size(3);

  // -- decl helpers --
  bool eq;
  int index;

  // -- get indices --
  int _index = num_per_thread*(threadIdx.x + blockDim.x * blockIdx.x);
  int ibatch = blockIdx.y;
  int ihead = blockIdx.z;

  for (int _ix = 0; _ix < num_per_thread; _ix++){

    // -- select (qi,ki) --
    index = _index + _ix;
    int qi = index / Kagg;
    int ki = index - qi*Kagg;

    for (int ks=0; ks < Ksearch; ks++){

      // -- find matching index --
      eq = true;
#pragma unroll
      for (int _idx=0; _idx < 3; _idx++){
        eq = eq and (fabs(inds[ibatch][ihead][qi][ki][_idx] -    \
                          qinds[ibatch][ihead][qi][ks][_idx]) < 1e-10);
      }

      // -- assign --
      if (eq){
#pragma unroll
        for (int _idx=0; _idx < 3; _idx++){
          grad_qinds[ibatch][ihead][qi][ks][_idx] = \
            grad_inds[ibatch][ihead][qi][ki][_idx]; // should be unique
        }
        continue; // pick next (qi,ki)
      }

    }
  }

} // fxn

void ref_bwd_inds_cuda(
    torch::Tensor grad_qinds, const torch::Tensor grad_inds,
    const torch::Tensor qinds, const torch::Tensor inds){

   // -- shape --
   int nbatch = inds.size(0);
   int nheads = inds.size(1);
   int nqueries = inds.size(2);
   int kagg = inds.size(3);


   // -- num threads --
   int _nthreads = 256;
   dim3 nthreads(_nthreads);

   // -- num blocks --
   int num_per_thread = 1;
   int nRun = nqueries*kagg;
   int _nblocks = (nRun-1)/(_nthreads*num_per_thread)+1;
   dim3 nblocks(_nblocks,nbatch,nheads);


   // -- launch kernel --
   AT_DISPATCH_FLOATING_TYPES(inds.type(),"ref_bwd_inds_kernel", ([&] {
   ref_bwd_inds_kernel<scalar_t><<<nblocks, nthreads>>>(
          grad_qinds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          grad_inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          qinds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          num_per_thread);
       }));

}

