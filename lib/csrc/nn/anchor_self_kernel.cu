
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <assert.h>
#include <cuda/std/type_traits>
#include "shared_nn_utils.cu"
// #include "search/shared_kernel.cu"

__device__ __forceinline__
void set_time_range(int& t_max, int ti, int T, int wt){
  int t_shift = min(0,ti - wt) + max(0,ti + wt - (T-1));
  // t_min = max(ti - wt - t_shift,0);
  t_max = min(T-1,ti + wt - t_shift);
}

// template< class T, class U >
// inline constexpr bool is_same_v = cuda::std::is_same<T, U>::value;
// at::ScalarType get_type_2(torch::Tensor my_tensor);
  

template <typename scalar_t, typename itype>
__global__ void anchor_self_kernel(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<itype,4,torch::RestrictPtrTraits> inds,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> order,
    int qstart, int stride0, int H, int W, int nHW, int nW, int q_per_thread){

  // -- starting qi for thread --
  int Q = dists.size(1);
  int K = dists.size(2);
  int bi = blockIdx.y;
  int qi_thread = q_per_thread*(threadIdx.x + blockDim.x * blockIdx.x);
  int self_index = 0;
  bool eq_loc;
  int iloc[3];
  itype loc[3];
  itype i_tmp[3];
  scalar_t d_tmp;
  int qi,i_mod,qindex;
  scalar_t delta,dmin_curr;
  int min_idx;

  // -- for each location --
  for (int qi_ix = 0; qi_ix < q_per_thread; qi_ix++){


    // -- current query --
    qi = qi_thread + qi_ix;
    if (qi >= Q){ continue; }
    qindex = qi + qstart;

    // -- unpack pixel locs --
    // get_pixel_loc(loc,  qi, tmp,  stride0, nW, nHW, H,W);
    int tmp = qindex;
    iloc[0] = qindex / nHW;
    tmp = (tmp - iloc[0]*nHW); 
    int nH_index = tmp / nW;
    iloc[1] = (nH_index*stride0) % H;
    tmp = tmp - nH_index*nW;
    iloc[2] = ((tmp % nW) * stride0) % W;

    // -- convert to type --
    if (is_same_v<itype,int>){
#pragma unroll
      for (int _idx = 0; _idx < 3; _idx++){
          loc[_idx] = iloc[_idx];
      }
    }else{
#pragma unroll
      for (int _idx = 0; _idx < 3; _idx++){
        loc[_idx] = __int2float_rn(iloc[_idx]);
      }
    }
    

    // -- search for matching index --
    min_idx = 0;
    dmin_curr = 10000;
    for (self_index = 0; self_index < K; self_index++){

      delta = 0;
      eq_loc = true;
      for (int ix=0; ix<3; ix++){
        if (is_same_v<itype,int>){
          eq_loc = eq_loc && (inds[bi][qi][self_index][ix] ==  loc[ix]);
        }else{
          delta += fabs(inds[bi][qi][self_index][ix] - loc[ix]);
        }
      }
      eq_loc = eq_loc && (delta < 1e-8);

      if (is_same_v<itype,int>){
        if (eq_loc){ min_idx = self_index; break; }
      }else{
        if (delta < 1e-8){ min_idx = self_index; break; }// break if equal
        else if (delta < dmin_curr){ // update min otherwise
          min_idx = self_index;
          dmin_curr = delta;
        }
      }
          
    }
    assert(min_idx<K);

    // -- swap dists --
    self_index = min_idx;
    d_tmp = dists[bi][qi][0];
    dists[bi][qi][0] = dists[bi][qi][self_index];
    dists[bi][qi][self_index] = d_tmp;
    order[bi][qi] = self_index;

    // -- swap inds --
#pragma unroll
    for(int ix=0; ix<3; ix++){
      i_tmp[ix] = inds[bi][qi][0][ix];
    }
#pragma unroll
    for(int ix=0; ix<3; ix++){
      inds[bi][qi][0][ix] = loc[ix];
    }
#pragma unroll
    for(int ix=0; ix<3; ix++){
      inds[bi][qi][self_index][ix] = i_tmp[ix];
    }
    
  }
}


void anchor_self_forward_cuda(
     torch::Tensor dists,
     torch::Tensor inds,
     torch::Tensor order,
     int qstart, int stride0, int H, int W){
  
  // -- unpack --
  int B = dists.size(0);
  int Q = dists.size(1);
  int K = dists.size(2);

  // -- derivative --
  int nH = (H-1)/stride0+1;
  int nW = (W-1)/stride0+1;
  int nHW = nH*nW;

  // -- num 2 run --
  int nRun = Q;

  // -- kernel params --
  int q_per_thread = 1;
  int _nthreads = 256;
  // int _nthreads = 128;
  dim3 nthreads(_nthreads);
  int _nblocks = (nRun-1)/(_nthreads*q_per_thread)+1;
  dim3 nblocks(_nblocks,B);
  // fprintf(stdout,"nblocks,nthreads: %d,%d\n",_nblocks,_nthreads);
  // fprintf(stdout,"nH,nW,stride0: %d,%d,%d\n",nH,nW,stride0);

  // -- launch kernel --
  auto itype = get_type(inds);
  auto dtype = get_type(dists);
  if (itype == torch::kInt32){
  // if (std::is_same_v<inds.type(),std::int32_t>){
    AT_DISPATCH_FLOATING_TYPES(dists.type(), "anchor_self_kernel", ([&] {
         anchor_self_kernel<scalar_t,int><<<nblocks, nthreads>>>(
         dists.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
         inds.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
         order.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
         qstart, stride0, H, W, nHW, nW, q_per_thread);
        }));
  }else if (itype == dtype){
  // }else if(std::is_same_v<inds.type(),dists.type()>){
    AT_DISPATCH_FLOATING_TYPES(dists.type(), "anchor_self_kernel", ([&] {
         anchor_self_kernel<scalar_t,scalar_t><<<nblocks, nthreads>>>(
         dists.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
         inds.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
         order.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
         qstart, stride0, H, W, nHW, nW, q_per_thread);
        }));

  }else{
    std::cout << "Must have inds type be int or match dists.\n" << std::endl;
    assert(1==0);
  }
  
}

/*********************************************************


     Anchor Self Time


*********************************************************/


template <typename scalar_t, typename itype>
__global__ void anchor_self_time_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<itype,6,torch::RestrictPtrTraits> inds,
    torch::PackedTensorAccessor32<itype,7,torch::RestrictPtrTraits> flows,
    int wt, int stride0, int st_offset,
    int H, int W, int nHW, int nW, int q_per_thread){

  // -- starting qi for thread --
  int HD = dists.size(1);
  int HD_f = flows.size(1);
  int Q = dists.size(2);
  int W_t = dists.size(3);
  int K = dists.size(4);
  int T = flows.size(2);
  int bi = blockIdx.y;
  int hi = blockIdx.z;
  int hi_f = hi % HD_f;
  int raster_idx = threadIdx.x + blockDim.x * blockIdx.x;
  int qi_thread = raster_idx/W_t;
  int st_i = (raster_idx - qi_thread*W_t);
  qi_thread = q_per_thread*qi_thread;
  int self_index = 0;
  bool eq_loc;
  int iloc[3];
  itype loc[3];
  itype i_tmp[3];
  scalar_t d_tmp;
  int qi,i_mod,qindex;
  scalar_t delta,dmin_curr;
  int min_idx;

  // -- for each location --
  if (st_i >= W_t ){ return; } 
  for (int qi_ix = 0; qi_ix < q_per_thread; qi_ix++){

    // -- current query --
    qi = qi_thread + qi_ix;
    if (qi >= Q){ continue; }
    qindex = qi;

    // -- unpack pixel locs --
    // get_pixel_loc(loc,  qi, tmp,  stride0, nW, nHW, H,W);
    int tmp = qindex;
    iloc[0] = qindex / nHW;
    tmp = (tmp - iloc[0]*nHW); 
    int nH_index = tmp / nW;
    iloc[1] = (nH_index*stride0) % H;
    tmp = tmp - nH_index*nW;
    iloc[2] = ((tmp % nW) * stride0) % W;

    // -- select time --
    int n_hi = iloc[1]/stride0;
    int n_wi = iloc[2]/stride0;
    int t_max;
    set_time_range(t_max, iloc[0], T, wt);
    int t_next = iloc[0] + st_i;
    t_next = (t_next > t_max) ? t_max - st_i : t_next;


    // -- get anchor index --
    loc[0] = t_next;
    if (st_i >= st_offset){
      auto flows_t = flows[bi][hi_f][iloc[0]][st_i-st_offset];
      loc[1] = iloc[1] + flows_t[1][n_hi][n_wi];
      loc[2] = iloc[2] + flows_t[0][n_hi][n_wi];
      loc[1] = bounds(loc[1],H);
      loc[2] = bounds(loc[2],W);
    }else{
      loc[1] = 1.*iloc[1];
      loc[2] = 1.*iloc[2];
    }

    // -- search for matching index --
    min_idx = 0;
    dmin_curr = 10000;
    for (self_index = 0; self_index < K; self_index++){

      delta = 0;
      eq_loc = true;
      for (int ix=0; ix<3; ix++){
        if (is_same_v<itype,int>){
          eq_loc = eq_loc && (inds[bi][hi][qi][st_i][self_index][ix] ==  loc[ix]);
        }else{
          delta += fabs(inds[bi][hi][qi][st_i][self_index][ix] - loc[ix]);
        }
      }
      eq_loc = eq_loc && (delta < 1e-8);

      if (is_same_v<itype,int>){
        if (eq_loc){ min_idx = self_index; break; }
      }else{
        if (delta < 1e-8){ min_idx = self_index; break; }// break if equal
        else if (delta < dmin_curr){ // update min otherwise
          min_idx = self_index;
          dmin_curr = delta;
        }
      }
          
    }
    assert(min_idx<K);

    // -- swap dists --
    self_index = min_idx;
    d_tmp = dists[bi][hi][qi][st_i][0];
    dists[bi][hi][qi][st_i][0] = dists[bi][hi][qi][st_i][self_index];
    dists[bi][hi][qi][st_i][self_index] = d_tmp;

    // -- swap inds --
#pragma unroll
    for(int ix=0; ix<3; ix++){
      i_tmp[ix] = inds[bi][hi][qi][st_i][0][ix];
    }
#pragma unroll
    for(int ix=0; ix<3; ix++){
      inds[bi][hi][qi][st_i][0][ix] = loc[ix];
    }
#pragma unroll
    for(int ix=0; ix<3; ix++){
      inds[bi][hi][qi][st_i][self_index][ix] = i_tmp[ix];
    }
    
  }
}


void anchor_self_time_forward_cuda(
     torch::Tensor dists, torch::Tensor inds,
     torch::Tensor flows, int wt, int stride0, int H, int W){
  
  // -- unpack --
  int B = dists.size(0);
  int HD = dists.size(1);
  int Q = dists.size(2);
  int W_t = dists.size(3);
  int K = dists.size(4);

  // -- derivative --
  int nH = (H-1)/stride0+1;
  int nW = (W-1)/stride0+1;
  int nHW = nH*nW;
  assert(W_t == 2*wt+1);
  int st_offset = W_t - flows.size(3);

  // -- num 2 run --
  int nRun = Q*W_t;

  // -- kernel params --
  int q_per_thread = 1;
  int _nthreads = 512;
  dim3 nthreads(_nthreads);
  int _nblocks = (nRun-1)/(_nthreads*q_per_thread)+1;
  dim3 nblocks(_nblocks,B,HD);
  // fprintf(stdout,"nblocks,nthreads: %d,%d\n",_nblocks,_nthreads);
  // fprintf(stdout,"nH,nW,stride0: %d,%d,%d\n",nH,nW,stride0);

  // -- launch kernel --
  auto itype = get_type(inds);
  auto dtype = get_type(dists);
  if (itype == torch::kInt32){
    fprintf(stdout,"int.\n");
    AT_DISPATCH_FLOATING_TYPES(dists.type(), "anchor_self_time_kernel", ([&] {
         anchor_self_time_kernel<scalar_t,int><<<nblocks, nthreads>>>(
         dists.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
         inds.packed_accessor32<int,6,torch::RestrictPtrTraits>(),
         flows.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
         wt, stride0, st_offset, H, W, nHW, nW, q_per_thread);
        }));
  }else if (itype == dtype){
    AT_DISPATCH_FLOATING_TYPES(dists.type(), "anchor_self_time_kernel", ([&] {
         anchor_self_time_kernel<scalar_t,scalar_t><<<nblocks, nthreads>>>(
         dists.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
         inds.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
         flows.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
         wt, stride0, st_offset, H, W, nHW, nW, q_per_thread);
        }));

  }else{
    std::cout << "Must have inds type be int or match dists.\n" << std::endl;
    assert(1==0);
  }
  
}



/*********************************************************


     Anchor Self Refine


*********************************************************/

template <typename scalar_t, typename itype>
__global__ void anchor_self_refine_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<itype,6,torch::RestrictPtrTraits> inds,
    torch::PackedTensorAccessor32<itype,5,torch::RestrictPtrTraits> flows,
    int stride0, int H, int W, int nHW, int nW, int q_per_thread){

  // -- starting qi for thread --
  int HD = dists.size(1);
  int HD_f = flows.size(1);
  int Q = dists.size(2);
  int G = dists.size(3);
  int K = dists.size(4);
  int bi = blockIdx.y;
  int hi = blockIdx.z;
  int hi_f = hi % HD_f;
  int raster_idx = threadIdx.x + blockDim.x * blockIdx.x;
  int qi_thread = raster_idx/G;
  int gi = (raster_idx - qi_thread*G);
  qi_thread = q_per_thread*qi_thread;
  int self_index = 0;
  bool eq_loc;
  itype loc[3];
  itype i_tmp[3];
  scalar_t d_tmp;
  int qi,i_mod;
  scalar_t delta,dmin_curr;
  int min_idx;

  // -- for each location --
  if (gi >= G ){ return; } 
  for (int qi_ix = 0; qi_ix < q_per_thread; qi_ix++){

    // -- current query --
    qi = qi_thread + qi_ix;
    if (qi >= Q){ continue; }


    // -- unpack pixel locs --
    loc[0] = round(flows[bi][hi_f][qi][gi][0]);
    loc[1] = flows[bi][hi_f][qi][gi][1];
    loc[2] = flows[bi][hi_f][qi][gi][2];

    // -- search for matching index --
    min_idx = 0;
    dmin_curr = 10000;
    for (self_index = 0; self_index < K; self_index++){

      delta = 0;
      eq_loc = true;
      for (int ix=0; ix<3; ix++){
        if (is_same_v<itype,int>){
          eq_loc = eq_loc && (inds[bi][hi][qi][gi][self_index][ix] ==  loc[ix]);
        }else{
          delta += fabs(inds[bi][hi][qi][gi][self_index][ix] - loc[ix]);
        }
      }
      eq_loc = eq_loc && (delta < 1e-8);

      if (is_same_v<itype,int>){
        if (eq_loc){ min_idx = self_index; break; }
      }else{
        if (delta < 1e-8){ min_idx = self_index; break; }// break if equal
        else if (delta < dmin_curr){ // update min otherwise
          min_idx = self_index;
          dmin_curr = delta;
        }
      }
          
    }
    assert(min_idx<K);

    // -- swap dists --
    self_index = min_idx;
    d_tmp = dists[bi][hi][qi][gi][0];
    dists[bi][hi][qi][gi][0] = dists[bi][hi][qi][gi][self_index];
    dists[bi][hi][qi][gi][self_index] = d_tmp;

    // -- swap inds --
#pragma unroll
    for(int ix=0; ix<3; ix++){
      i_tmp[ix] = inds[bi][hi][qi][gi][0][ix];
    }
#pragma unroll
    for(int ix=0; ix<3; ix++){
      inds[bi][hi][qi][gi][0][ix] = loc[ix];
    }
#pragma unroll
    for(int ix=0; ix<3; ix++){
      inds[bi][hi][qi][gi][self_index][ix] = i_tmp[ix];
    }
  }

}


void anchor_self_refine_forward_cuda(
     torch::Tensor dists, torch::Tensor inds,
     torch::Tensor flows,
     int stride0, int H, int W){
  
  // -- unpack --
  int B = dists.size(0);
  int HD = dists.size(1);
  int Q = dists.size(2);
  int G = dists.size(3);
  int K = dists.size(4);

  // -- derivative --
  int nH = (H-1)/stride0+1;
  int nW = (W-1)/stride0+1;
  int nHW = nH*nW;

  // -- num 2 run --
  int nRun = Q*G;

  // -- kernel params --
  int q_per_thread = 1;
  int _nthreads = 512;
  dim3 nthreads(_nthreads);
  int _nblocks = (nRun-1)/(_nthreads*q_per_thread)+1;
  dim3 nblocks(_nblocks,B,HD);
  // fprintf(stdout,"nblocks,nthreads: %d,%d\n",_nblocks,_nthreads);
  // fprintf(stdout,"nH,nW,stride0: %d,%d,%d\n",nH,nW,stride0);

  // -- launch kernel --
  auto itype = get_type(inds);
  auto dtype = get_type(dists);
  if (itype == torch::kInt32){
    fprintf(stdout,"int.\n");

    AT_DISPATCH_FLOATING_TYPES(dists.type(), "anchor_self_refine_kernel", ([&] {
         anchor_self_refine_kernel<scalar_t,int><<<nblocks, nthreads>>>(
         dists.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
         inds.packed_accessor32<int,6,torch::RestrictPtrTraits>(),
         flows.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
         stride0, H, W, nHW, nW, q_per_thread);
        }));
  }else if (itype == dtype){
    AT_DISPATCH_FLOATING_TYPES(dists.type(), "anchor_self_refine_kernel", ([&] {
         anchor_self_refine_kernel<scalar_t,scalar_t><<<nblocks, nthreads>>>(
         dists.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
         inds.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
         flows.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
         stride0, H, W, nHW, nW, q_per_thread);
        }));

  }else{
    std::cout << "Must have inds type be int or match dists.\n" << std::endl;
    assert(1==0);
  }
  
}

