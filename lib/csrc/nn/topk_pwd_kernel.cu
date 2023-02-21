
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <assert.h>
// #include "shared_nn_utils.cu"
#include "../search/shared_kernel.cu"


template <typename scalar_t>
__global__ void topk_pwd_kernel(
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds0,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds1,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dists,
    int ps, int pt, int dilation, bool reflect_bounds, 
    bool use_adj, int off_H0, int off_W0, int off_H1, int off_W1,
    int q_per_thread, int pwd_per_thread, int nblocks_q, int nblocks_pwd){

  // -- unpack shape --
  int B = vid.size(0);
  int HD = vid.size(1);
  int T = vid.size(2);
  int C = vid.size(3);
  int H = vid.size(4);
  int W = vid.size(5);
  int Q = dists.size(2);
  int nPWD = dists.size(3);
  int K = inds0.size(3);

  // invalid value 
  float invalid = __int_as_float(0x7f800000);

  // -- search region offsets --
  int psHalf = (ps)/2;
  int adj = use_adj ? psHalf : 0;

  // -- cuda index --
  int ibatch = blockIdx.x;
  int ihead = blockIdx.y;
  int raster = blockIdx.z;
  int q_block = raster / nblocks_pwd;
  int pwd_block = (raster - q_block*nblocks_pwd);
  int qi_start = (threadIdx.x + q_block*blockDim.x)*q_per_thread;
  int qi_end = min(qi_start+q_per_thread,Q);
  int pwd_start = (threadIdx.y + pwd_block*blockDim.y)*pwd_per_thread;
  int pwd_end = min(pwd_start+pwd_per_thread,nPWD);
  int qi,pwd_i;

  // decls
  int patch0[3];
  int patch1[3];
  int pix0[3];
  int pix1[3];
  bool valid;
  bool valid0[4];
  bool valid1[4];
  scalar_t dist,_pix0,_pix1,_dist;

  // 1-dim -> 2-dim
  int ki_0,ki_1;
  float xform_p;
  int xform_i0;
  int xform_i0p;
  int xform_tmp;
  bool xform_bool;

  // -- cleaner code --
  int center_offsets[4] = {off_H0,off_H1,off_W0,off_W1};
  int patch_offset = psHalf + adj;

  // -- for each location --
  for (int qi = qi_start; qi < qi_end; qi++){
    for (int pwd_i = pwd_start; pwd_i < pwd_end; pwd_i++){

      // -- get pair of k's --
      // --> thank you below <--
      // https://atrebas.github.io/post/2021-01-17-index_to_lower_triangular_subscripts/
      xform_p = (sqrtf(1.+8.*(pwd_i))-1.)/2.;
      xform_i0 = __float2int_rd(xform_p);
      xform_i0p = xform_i0 + 1;
      xform_tmp = pwd_i-xform_i0*(xform_i0+1)/2;
      xform_bool = xform_p == xform_i0;
      ki_0 = xform_i0p;
      ki_1 = xform_tmp;
      // ki_0 = xform_bool ? xform_i0 : xform_i0p;
      // ki_1 = xform_bool ? xform_i0 : xform_tmp;
      // ki_0 -= 1;
      // ki_1 -= 1;

      // -- fill locally --
      #pragma unroll
      for (int i=0; i < 3; i++){
        patch0[i] = inds0[ibatch][ihead][qi][ki_0][i];
        patch1[i] = inds1[ibatch][ihead][qi][ki_1][i];
      }

      // -- compute dist --
      dist = 0;
      compute_dist<scalar_t,1>(dist,
                   vid[ibatch][ihead],vid[ibatch][ihead],
                   patch0, patch1, pix0, pix1, valid0, valid1,
                   ps,pt,dilation,reflect_bounds,
                   patch_offset,center_offsets,invalid,
                   T,C,H,W,_pix0,_pix1,_dist);

      // -- assign --
      dists[ibatch][ihead][qi][pwd_i] = dist;
      
    }
  }
}


void topk_pwd_forward_cuda(const torch::Tensor vid,
    const torch::Tensor inds0, const torch::Tensor inds1,
    torch::Tensor dists, int ps, int pt,
    int dilation, bool reflect_bounds, bool use_adj,
    int off_H0, int off_W0, int off_H1, int off_W1){

  // -- unpack --
  int B = inds0.size(0);
  int HD = inds0.size(1);
  int Q = inds0.size(2);
  int K = inds0.size(3);
  int nPWD = dists.size(3);
  int _nPWD = K*(K-1)/2;
  assert(nPWD == _nPWD);

  // -- kernel params --
  int q_per_thread = 1;
  int pwd_per_thread = 1;
  int nthreads_q = 32;
  int nthreads_pwd = 32;
  int nblocks_q = (Q-1)/(nthreads_q*q_per_thread)+1;
  int nblocks_pwd = (nPWD-1)/(nthreads_pwd*pwd_per_thread)+1;
  int _nblocks = nblocks_q * nblocks_pwd;
  dim3 nthreads(nthreads_q,nthreads_pwd);
  dim3 nblocks(B,HD,_nblocks);
  // fprintf(stdout,"ps: %d\n",ps);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(vid.type(),"non_local_search_forward_kernel", ([&] {
        topk_pwd_kernel<scalar_t><<<nblocks, nthreads>>>(
          vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          inds0.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          inds1.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          ps, pt, dilation, reflect_bounds, use_adj,
          off_H0, off_W0, off_H1, off_W1,
          q_per_thread, pwd_per_thread, nblocks_q, nblocks_pwd);
      }));
}
