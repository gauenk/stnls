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


