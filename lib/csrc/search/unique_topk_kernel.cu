
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
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> vals,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> args,
    int k, int dim, int qpt){

  // shapes
  int a = vals.size(0);
  int b = vals.size(1);
  int nblocks = (dim == 1) ? a : b;
  int ki = threadIdx.x;
  float dist;
  int qi,ai,nuniq;

  // -- iterate over blocks --
  for(int _qpt = 0; _qpt < qpt; _qpt++){
    qi = _qpt + blockIdx.x*qpt;
    if (qi >= nblocks){ continue; }

    // -- init --
    ai = 0;
    nuniq = 0;
    
    // -- get starting location for each thread --
    for (int _ki = 0; _ki < b; _ki++){
      dist = fabsf(vals[qi][ai] - vals[qi][_ki]);
      ai = ((dist > 1e-10) && (nuniq < ki)) ? _ki : ai;
      nuniq += (dist > 1e-10);
    }

    // -- line-up --
    __syncthreads();

    // -- write --
    args[qi][ki] = ai;
    
  }
}

void unique_topk_forward_cuda(torch::Tensor vals,
                              torch::Tensor args,
                              int k, int dim){

   // -- comp per threads --
   int a = vals.size(0);
   int b = vals.size(1);
   int num_per_thread = 2;
   int nblocks = (dim == 1) ? a : b;
   nblocks = (nblocks-1)/num_per_thread+1;

   // -- num threads --
   dim3 nthreads(k);

   // fprintf(stdout,"nthreads_k0: %d\n",nthreads_k0);
   // fprintf(stdout,"nbatch,nheads,nquery_blocks: %d,%d,%d\n",
   //         nbatch,nheads,nquery_blocks);
   // fprintf(stdout,"qpt,nquery_blocks,w_threads: %d,%d,%d,%d\n",
   //         qpt,nquery_blocks,ws_h_threads,ws_w_threads);
   // fprintf(stdout,"reflect_bounds,search_abs,anchor_self: %d,%d,%d\n",
   //         reflect_bounds,search_abs,anchor_self);
    
   // launch kernel
   AT_DISPATCH_FLOATING_TYPES(vals.type(),
                              "unique_topk_forward_kernel", ([&] {
      unique_topk_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        vals.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        args.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
        k, dim, num_per_thread);
      }));
}


/****************************

       Backward Pass

****************************/

// none yet