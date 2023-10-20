#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include <cuda/std/type_traits>
#include <cstdio>
#include <math.h>
#include <ATen/ATen.h>

template< class T, class U >
inline constexpr bool is_same_v = cuda::std::is_same<T, U>::value;
using namespace at;

__inline__ __device__ int bounds(int val, int lim ){
  int vval = val;
  if (val < 0){
    vval = -val;
  }else if (val >= lim){
    vval = 2*(lim-1) - val;
  }
  return vval;
}

__inline__ int cpu_bounds(int val, int lim ){
  int vval = val;
  if (val < 0){
    vval = -val;
  }else if (val >= lim){
    vval = 2*(lim-1) - val;
  }
  return vval;
}

template<typename itype=int>
__device__ __forceinline__ 
void get_query_loc(itype* pix,  int qindex, int nW0, int nHW0){
  int tmp,nH_index;
  if (is_same_v<itype,int>){
    tmp = qindex;
    pix[0] = tmp / nHW0;
    tmp = (tmp - pix[0]*nHW0); 
    nH_index = tmp / nW0;
    pix[1] = nH_index;
    tmp = tmp - nH_index*nW0;
    pix[2] = tmp % nW0;
  }else{
    tmp = qindex;
    pix[0] = floor(tmp/nHW0);
    tmp = (tmp - pix[0]*nHW0); 
    nH_index = tmp / nW0;
    pix[1] = floor(nH_index);
    tmp = tmp - nH_index*nW0;
    pix[2] = floor(tmp % nW0);
  }
}


template<typename scalar_t>
__device__ __forceinline__ 
void bilin2d_interpolate(scalar_t& pix, scalar_t hi, scalar_t wi, int H, int W,
     const torch::TensorAccessor<scalar_t,2,torch::RestrictPtrTraits,int32_t> vid){

  // -- interpolated locations --
  int h_interp,w_interp;
  scalar_t w;

  // -- interpolate pixel value --
  pix = 0;
#pragma unroll
  for (int ix=0;ix<2;ix++){
#pragma unroll
    for (int jx=0;jx<2;jx++){

      // -- interpolation weight --
      h_interp = __float2int_rz(hi+ix);
      w = max(0.,1-fabs(h_interp-hi));
      w_interp = __float2int_rz(wi+jx);
      w = w*max(0.,1-fabs(w_interp-wi));

      // -- ensure legal bounds --
      h_interp = bounds(h_interp,H);
      w_interp = bounds(w_interp,W);

      // -- update --
      pix += w*vid[h_interp][w_interp];
    }
  }

}
