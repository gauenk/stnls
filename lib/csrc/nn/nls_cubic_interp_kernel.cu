
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <assert.h>
#include <cuda/std/type_traits>
#include "shared_nn_utils.cu"

// template< class T, class U >
// inline constexpr bool is_same_v = cuda::std::is_same<T, U>::value;
// at::ScalarType get_type_2(torch::Tensor my_tensor);
  

template <typename scalar_t>
__global__ void nls_cubic_interp_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> inds,
    int q_per_thread){

  // -- starting qi for thread --
  int Q = dists.size(1);
  int st = dists.size(2);
  int ws_h = dists.size(3);
  int ws_w = dists.size(4);
  int bi = blockIdx.y;
  int qi_thread = q_per_thread*(threadIdx.x + blockDim.x * blockIdx.x);
  int self_index = 0;
  bool eq_loc;
  int iloc[3];
  itype loc[3];
  itype i_tmp[3];
  scalar_t d_tmp;
  int qi,i_mod,_qi;
  scalar_t delta,dmin_curr;
  int min_idx;

  scalar_t vals[4];
  scalar_t coeff[4];
  scalar_t quad[3];
  int dx = 0;
  scalar_t discr,discr_sq;
  scalar_t root;
  scalar_t eval;
  scalar_t curr_val,final_val;
  float inf = __int_as_float(0x7f800000);
  bool use_interp;
  scalar_t inds_i[3];
  int wi[4];
  int wj[4];

  // -- spatial locations --
  #pragma unroll
  for(int _ix = 0; _ix < 4; _ix++){
    for(int _jx = 0; _jx < 4; _jx++){
      wi[_ix] = threadIdx.x + _ix;
      wj[_ix] = threadIdx.y + +ix;
    }
  }

  // -- for each location --
  for (int qi_ix = 0; qi_ix < q_per_thread; qi_ix++){

    // -- current query --
    _qi = qi_thread + qi_ix;
    if (qi >= Q){ break; }
    qi = _qi + qstart;

    // -- across time --
    for (int ti = 0; ti < st; ti++){


      // -- read distance values --
#pragma unroll
      for (int _ix = 0; _ix < 4; _ix++){
        vals[_ix] = dists[ibatch][ihead][qi][ti][wi[_ix]][wj[_ix]];
      }
      curr_val = vals[1];

      // -- compute interp coefficients --
      coeff[0] = vals[3] - vals[2] - vals[1] - vals[0];
      coeff[1] = vals[0] - vals[1] - coeff[0];
      coeff[2] = vals[2] - vals[0];
      coeff[3] = vals[1];

      // -- compute poly coeff --
      quad[0] = 3 * coeff[0];
      quad[1] = 2 * coeff[1];
      quad[2] = coeff[2];
    
      // -- compute roots --
      discr = __powf(quad[1],2) - 4 * quad[0] * quad[2];
      discr_sq = discr > 0 ? sqrt((float)discr) : -inf;
      if (DIST_TYPE == 0){ // prod
        root = discr > 0 ? (-quad[0] + discr_sq)/(2*quad[0]) : -inf; // bigger better
      }else{ // l2
        root = discr > 0 ? (-quad[0] - discr_sq)/(2*quad[0]) : inf; // smaller better
      }
    
      // -- evaluate roots --
      eval = coeff[0] * __powf(root,3) + coeff[1] * __powf(root,2) \
        + coeff[2] * root + coeff[3];
    
      // -- select dists [root, curr_val] --
      if (DIST_TYPE == 0){ // prod
        use_interp = eval > curr_val; // bigger better
      }else{ // l2
        use_interp = eval < curr_val; // smaller better
      }
      final_val = use_interp ? eval : curr_val;
    
      // -- linearly interp inds --
#pragma unroll
      for (int _ix=1; _ix < 3; _ix++){
        inds_i[_ix] = root * inds_i[l0][_ix]  + (1-root) * inds_i[l1][_ix];
      }

      // -- sync all threads --
      __synchronize();

      // -- write --
      dists[ibatch][ihead][qi][_ix+1] = final_val;
      // inds[ibatch][ihead][qi][st][wi][wj][0] = inds_i[0];
      inds[ibatch][ihead][qi][st][wi][wj][1] = inds_i[1];
      inds[ibatch][ihead][qi][st][wi][wj][2] = inds_i[2];

    }
  }
}


void nls_cubic_interp_forward_cuda(
     torch::Tensor dists,
     torch::Tensor inds,
     int qstart, int stride0, int H, int W){
  
  // -- unpack --
  int B = dists.size(0);
  int Q = dists.size(1);
  int K = dists.size(2);

  // -- derivative --
  int nH = (H-1)/stride0+1;
  int nW = (W-1)/stride0+1;
  int nHW = nH*nW;


   // -- derived quantities --
   int H = vid0.size(4);
   int W = vid0.size(5);
   int nH0 = (H-1)/stride0+1;
   int nW0 = (W-1)/stride0+1;
   int nHW0 = nH0 * nW0;

   // -- threads --
   int nheads = dists.size(1);
   int nqueries = dists.size(2);
   int st = dists.size(3);
   int ws_h = dists.size(4);
   int ws_w = dists.size(5);
   assert (ws_h <= 31);
   assert (ws_w <= 31);
   int ws_h_threads = std::min(ws_h,31);
   int ws_w_threads = std::min(ws_w,31);
   int ws_h_per_thread = ((ws_h-1)/ws_h_threads) + 1;
   int ws_w_per_thread = ((ws_w-1)/ws_w_threads) + 1;
   dim3 nthreads(ws_h_threads,ws_w_threads);

   // -- nblocks --
   int B = vid0.size(0);
   int HD = vid0.size(1);
   int q_per_thread = 2;
   int nquery_blocks = ((nqueries - 1) / q_per_thread) + 1;
   dim3 nblocks(nquery_blocks,B,HD);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(dists.type(), "nls_cubic_interp_kernel", ([&] {
         nls_cubic_interp_kernel<scalar_t><<<nblocks, nthreads>>>(
         dists.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
         inds.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
         q_per_thread);
      }));
  
}
