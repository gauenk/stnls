
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
  

template <typename scalar_t, typename itype>
__global__ void anchor_self_kernel(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<itype,4,torch::RestrictPtrTraits> inds,
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
    for (self_index = 0; self_index < K; self_index++){

      eq_loc = true;
      for (int ix=0; ix<3; ix++){
        if (is_same_v<itype,int>){
          eq_loc = eq_loc && (inds[bi][qi][self_index][ix] ==  loc[ix]);
        }else{
          eq_loc = eq_loc && (fabs(inds[bi][qi][self_index][ix] - loc[ix]) < 1e-8);
        }
      }
      if (eq_loc){ break; }
    }
    assert(self_index<K);

    // -- swap dists --
    d_tmp = dists[bi][qi][0];
    dists[bi][qi][0] = dists[bi][qi][self_index];
    dists[bi][qi][self_index] = d_tmp;

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
         qstart, stride0, H, W, nHW, nW, q_per_thread);
        }));
  }else if (itype == dtype){
  // }else if(std::is_same_v<inds.type(),dists.type()>){
    AT_DISPATCH_FLOATING_TYPES(dists.type(), "anchor_self_kernel", ([&] {
         anchor_self_kernel<scalar_t,scalar_t><<<nblocks, nthreads>>>(
         dists.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
         inds.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
         qstart, stride0, H, W, nHW, nW, q_per_thread);
        }));

  }else{
    std::cout << "Must have inds type be int or match dists.\n" << std::endl;
    assert(1==0);
  }
  
}

// at::ScalarType get_type_2(torch::Tensor my_tensor){
//   const auto& the_type = my_tensor.type();
//   at::ScalarType _st = ::detail::scalar_type(the_type);
//   return _st;
// }

/************************************************

            With Ordering

************************************************/


// template <typename scalar_t, typename itype>
// __global__ void anchor_self_with_ordering_kernel(
//     torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> dists,
//     torch::PackedTensorAccessor32<itype,4,torch::RestrictPtrTraits> inds,
//     torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> order,
//     int qstart, int stride0, int H, int W, int nHW, int nW, int q_per_thread){

//   // -- starting qi for thread --
//   int Q = dists.size(1);
//   int K = dists.size(2);
//   int bi = blockIdx.y;
//   int qi_thread = q_per_thread*(threadIdx.x + blockDim.x * blockIdx.x);
//   int self_index = 0;
//   bool eq_loc;
//   int iloc[3];
//   itype iloc[3];
//   itype i_tmp[3];
//   int o_tmp[3];
//   scalar_t d_tmp;
//   int qi,i_mod,qindex;

//   // -- for each location --
//   for (int qi_ix = 0; qi_ix < q_per_thread; qi_ix++){

//     // -- current query --
//     qi = qi_thread + qi_ix;
//     if (qi >= Q){ continue; }
//     qindex = qi + qstart;

//     // -- unpack pixel locs --
//     int tmp = qindex;
//     iloc[0] = qindex / nHW;
//     tmp = (tmp - loc[0]*nHW); 
//     int nH_index = tmp / nW;
//     iloc[1] = (nH_index*stride0) % H;
//     tmp = tmp - nH_index*nW;
//     iloc[2] = ((tmp % nW) * stride0) % W;

//     // -- convert to type --
//     if (~is_same_v<itype,int>){
// #pragma unroll
//       for (int _idx = 0; _idx < 3; _idx++){
//           loc[_idx] = __int2float_rn(loc[_idx]);
//       }
//     }

//     // -- search for matching index --
//     for (self_index = 0; self_index < K; self_index++){

//       eq_loc = true;
//       for (int ix=0; ix<3; ix++){
//         if (is_same_v<itype,int>){
//           eq_loc = eq_loc && (inds[bi][qi][self_index][ix] ==  loc[ix]);
//         }else{
//           eq_loc = eq_loc && fabs((inds[bi][qi][self_index][ix]) - loc[ix]) < 1e-8;
//         }
//       }
//       if (eq_loc){ break; }
//     }
//     assert(self_index<K);

//     // -- swap dists --
//     d_tmp = dists[bi][qi][0];
//     dists[bi][qi][0] = dists[bi][qi][self_index];
//     dists[bi][qi][self_index] = d_tmp;


//     // -- swap inds --
// #pragma unroll
//     for(int ix=0; ix<3; ix++){
//       i_tmp[ix] = inds[bi][qi][0][ix];
//       o_tmp[ix] = order[bi][qi][0][ix];
//     }
// #pragma unroll
//     for(int ix=0; ix<3; ix++){
//       inds[bi][qi][0][ix] = loc[ix];
//       order[bi][qi][0][ix] = order[bi][qi][self_index][ix];
//     }
// #pragma unroll
//     for(int ix=0; ix<3; ix++){
//       inds[bi][qi][self_index][ix] = i_tmp[ix];
//       order[bi][qi][self_index][ix] = o_tmp[ix];
//     }
    
//   }
// }


// void anchor_self_with_ordering_forward_cuda(
//      torch::Tensor dists,
//      torch::Tensor inds,
//      int qstart, int stride0, int H, int W){
  
//   // -- unpack --
//   int B = dists.size(0);
//   int Q = dists.size(1);
//   int K = dists.size(2);

//   // -- derivative --
//   int nH = (H-1)/stride0+1;
//   int nW = (W-1)/stride0+1;
//   int nHW = nH*nW;

//   // -- num 2 run --
//   int nRun = Q;

//   // -- kernel params --
//   int q_per_thread = 1;
//   int _nthreads = 256;
//   dim3 nthreads(_nthreads);
//   int _nblocks = (nRun-1)/(_nthreads*q_per_thread)+1;
//   dim3 nblocks(_nblocks,B);
//   // fprintf(stdout,"nblocks,nthreads: %d,%d\n",_nblocks,_nthreads);
//   // fprintf(stdout,"nH,nW,stride0: %d,%d,%d\n",nH,nW,stride0);

//   // -- launch kernel --
//   auto itype = get_type(inds);
//   auto dtype = get_type(dists);
//   if (itype == torch::kInt32){
//   // if (std::is_same_v<inds.type(),std::int32_t>){
//     AT_DISPATCH_FLOATING_TYPES(dists.type(),
//                                "anchor_self_with_ordering_kernel", ([&] {
//          anchor_self_with_ordering_kernel<scalar_t,int><<<nblocks, nthreads>>>(
//          dists.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//          inds.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
//          order.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//          qstart, stride0, H, W, nHW, nW, q_per_thread);
//         }));
//   }else if (itype == dtype){
//   // }else if(std::is_same_v<inds.type(),dists.type()>){
//     AT_DISPATCH_FLOATING_TYPES(dists.type(),
//                                "anchor_self_with_ordering_kernel", ([&] {
//          anchor_self_with_ordering_kernel<scalar_t,scalar_t><<<nblocks, nthreads>>>(
//          dists.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//          inds.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//          order.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//          qstart, stride0, H, W, nHW, nW, q_per_thread);
//         }));

//   }else{
//     std::cout << "Must have inds type be int or match dists.\n" << std::endl;
//     assert(1==0);
//   }
  
// }

