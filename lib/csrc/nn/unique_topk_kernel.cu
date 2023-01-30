
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
using namespace at;

/****************************

       Inline Functions

****************************/

__inline__ __device__ int bounds(int val, int lim ){
  int vval = val;
  if (val < 0){
    vval = -val;
  }else if (val >= lim){
    vval = 2*(lim-1) - val;
  }
  return vval;
}

inline __host__ __device__
int unravel_index(int& ti, int& hi, int& wi, const int qindex,
                  const int h, const int w, const int hw){
  // index to pixel location
  int i_mod = qindex % hw;
  ti = qindex / hw;
  wi = (i_mod % w);
  hi = (i_mod / w) % h;
}

inline __host__ __device__ int get_backward_window_start(const int index, const int KERNEL_SIZE, const int NEIGHBORHOOD_SIZE)
{
    return (index < KERNEL_SIZE) ? (0) : index - NEIGHBORHOOD_SIZE;
}


/****************************

       Forward Pass

****************************/

template <typename scalar_t>
__global__ void unique_topk_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dists,
    const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> inds,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dists_topk,
    torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> inds_topk,
    int k, int qpt){

  // -- shaping vars --
  int qi_cuda = threadIdx.x + blockDim.x * blockIdx.x;
  int Q = dists.size(0);
  int k_in = dists.size(1);

  // -- major index --
  int kj = 0;
  int qi = 0;

  // -- checking vars --
  int kj0 = 0;
  int diff;
  int kl;
  scalar_t dist_curr,dist_prev;
  bool any_same,reset_check;
  int check_start;
  dist_prev = (scalar_t)__int_as_float(0x7f800000);

  // -- iterate over blocks --
  for(int qi_ix = 0; qi_ix < qpt; qi_ix++){

    // -- specify query index --
    qi = qi_cuda + qi_ix;
    if (qi >= Q){ continue; }

    // -- init --
    kj0 = 0;
    check_start = 0;

    // -- find value for each "k" location --
    for (int ki = 0; ki < k; ki++){

      // -- terminate (& fail) if kj,ki too beyond limits --
      if (kj0 >= k_in){
        break;
      }

      // -- assign --
      dist_curr = dists[qi][kj0];
      dists_topk[qi][ki] = dist_curr;
      inds_topk[qi][ki][0] = inds[qi][kj0][0];
      inds_topk[qi][ki][1] = inds[qi][kj0][1];
      inds_topk[qi][ki][2] = inds[qi][kj0][2];

      // -- update "same dist" history --
      reset_check = fabs((float)(dist_curr - dist_prev))>1e-5;
      check_start = reset_check ? kj0 : check_start;
      dist_prev = dist_curr;

      // -- find next kj --
      for (kj = kj0+1; kj < k_in; kj++){

        // -- Keep moving "kj" forward until we find one that is different
        // -- than all previous "kj" indices with the same "dist" value.
        any_same = false;
        for (kl = check_start; kl <= kj0; kl++){

          diff = 0;
          #pragma unroll
          for (int ix = 0; ix < 3; ix++){
            diff += inds[qi][kj][ix] != inds[qi][kl][ix];
          }
          any_same = (diff == 0) || any_same;
        }

        // -- stop incrementing at unique kj --
        if (not any_same){
          break;
        }          

      }
      kj0 = kj; // update selected neighbor

    }
  }
}

void unique_topk_forward_cuda(const torch::Tensor dists, const torch::Tensor inds,
                              torch::Tensor dists_topk, torch::Tensor inds_topk,
                              int k){

   // -- comp per threads --
   int Q = dists.size(0);
   int K = dists.size(1);

   // -- blocks --
   int queries_per_thread = 1;
   int nthreads = 256;
   int nblocks = (Q-1)/(queries_per_thread*nthreads)+1;

   // launch kernel
   AT_DISPATCH_FLOATING_TYPES(dists.type(),"unique_topk_forward_kernel", ([&] {
      unique_topk_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        dists.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
        dists_topk.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        inds_topk.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
        k, queries_per_thread);
      }));
}


/****************************

       Backward Pass

****************************/

// none yet