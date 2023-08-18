
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuda/std/type_traits>
#include <cstdio>
// #include <cuda/std/type_traits>
// #include <cstdio>

inline
at::ScalarType get_type(const torch::Tensor my_tensor){
  const auto& the_type = my_tensor.type();
  at::ScalarType _st = ::detail::scalar_type(the_type);
  return _st;
}

template< class T, class U >
inline constexpr bool is_same_v = ::cuda::std::is_same<T, U>::value;

__inline__ __device__ int bounds(int val, int lim ){
  int vval = val;
  if (val < 0){
    vval = -val;
  }else if (val >= lim){
    vval = 2*(lim-1)-val;
  }
  return vval;
}

__device__ __forceinline__
bool check_interval(int val, int lower, int upper){
  return (val >= lower) && (val < upper);
}

__inline__ __device__ int bounds2(int val, int lb, int ub){
  int vval = val;
  if ((val < lb) && (lb > 0)){
    vval = 2*lb - val;
  }else if (val < 0){
    vval = -val;
  }else if (val >= ub){
    vval = 2*(ub-1)-val;
  }
  return vval;
}


