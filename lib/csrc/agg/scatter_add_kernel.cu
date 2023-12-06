/*

  Stack non-local patches into a video

*/

// #include "scatter_int.cu"

// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "../shared_kernel.cu"

/****************************

       Forward Pass

****************************/

template <typename scalar_t>
__global__ void scatter_add_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> out_vid,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> counts,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> in_vid,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dists,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
    int ps, int strideIn, int strideOut, int pt, int dilation, bool reflect_bounds,
    int patch_offset, int q_per_thread){

    // -- shapes --
    int B = in_vid.size(0);
    int HD = in_vid.size(1);
    int T = in_vid.size(2);
    int F = in_vid.size(3);
    int inH = in_vid.size(4);
    int inW = in_vid.size(5);
    int outH = out_vid.size(4);
    int outW = out_vid.size(5);
    int Q = inds.size(2);
    int K = inds.size(3);

    // -- batching --
    int query_start = (threadIdx.x + blockDim.x*blockIdx.x)*q_per_thread;
    // int query_start = blockIdx.x*blockDim.x+threadIdx.x;
    int ki = blockIdx.y*blockDim.y+threadIdx.y;
    int ihead = blockIdx.z/B;
    int ibatch = (blockIdx.z-ihead*B) % B;
    // int ibatch = blockIdx.y;
    // int ihead = blockIdx.z;
    if (ki >= K){ return; }

    // // -- cuda threads --
    // int pi = threadIdx.y;
    // int pj = threadIdx.z;

    // -- pixel locations --
    int qi;
    bool valid;
    scalar_t pix,weight;
    int ref_ti,nl_ti;
    int ref[3],ref_p[3],nl[3];
    // int nW = (outW-1)/strideOut+1;
    // int nHW = nW*((outH-1)/strideOut+1);
    int nW = (inW-1)/strideIn+1;
    int nHW = nW*((inH-1)/strideIn+1);

    // -- across queries --
    for(int _qi = 0; _qi < q_per_thread; _qi++){

      // -- query index --
      qi = query_start + _qi;
      if (qi >= Q){ continue; }

      // -- write location --
      get_pixel_loc<int>(wref,qi,strideIn,nW,nHW,inH,inW);

      // -- non-local index --
      get_pixel_loc<int>(ref,qi,strideOut,nW,nHW,outH,outW);
  #pragma unroll
      for (int _idx=0; _idx < 3; _idx++){
        nl[_idx] = ref[_idx] + inds[ibatch][ihead][qi][ki][_idx];
      }

      // -- check "inf" (but it won't be inf sometimes)  --
      valid = (abs(nl[1]) < 1e7) and (abs(nl[2]) < 1e7);
      if (not(valid)){ continue; }

      // -- always reflect anchor point --
      nl[0] = bounds(nl[0],T);
      nl[1] = bounds(nl[1],outH);
      nl[2] = bounds(nl[2],outW);

      // -- non-local weight --
      weight = dists[ibatch][ihead][qi][ki];

      // -- iterate over patches --
      for(int pi=0; pi < ps; pi++){
      for(int pj=0; pj < ps; pj++){

        // -- reference pixel index --
        ref_p[0] = ref[0];
        ref_p[1] = ref[1]+dilation*(pi + patch_offset);
        ref_p[2] = ref[2]+dilation*(pj + patch_offset);
        check_bounds(valid, ref_p, T,  inH, inW);
        if (not valid){ continue; }
  
        // -- increment legal refs --
        if ((ref[0]==0) and (ibatch==0) and (ihead==0) and (ki==0)){
          atomicAdd(&counts[ref_p[1]][ref_p[2]],1);
        }
  
        // -- non-local pixel index --
        nl[1] = nl[1]+dilation*(pi + patch_offset);
        nl[1] = reflect_bounds ? bounds(nl[1],outH) : nl[1];
        nl[2] = nl[2]+dilation*(pj + patch_offset);
        nl[2] = reflect_bounds ? bounds(nl[2],outW) : nl[2];
        check_bounds(valid, nl, T,  outH, outW);
        if (not valid){ continue; }

        // -- iterate over loop --
        for(int pk = 0; pk < pt; pk++){

          // -- time is always valid --
          ref_ti = ref_p[0] + pk;
          nl_ti = reflect_bounds ? bounds(nl[0]+pk,T) : (nl[0]+pk);
          valid = (nl_ti >= 0) && (nl_ti < T) and (ref_ti < T);
          if (not valid){ continue; }

          // -- channels --
          for(int iftr = 0; iftr < F; iftr++){

            // -- fill --
            pix = weight*in_vid[ibatch][ihead][ref_ti][iftr][ref_p[1]][ref_p[2]];
            atomicAdd(&out_vid[ibatch][ihead][nl_ti][iftr][nl[1]][nl[2]],pix);

          } // nfeatures-loop
        } // pt-loop
      }} // pi,pj
  } // query-loop
}

void scatter_add_forward_cuda(
    torch::Tensor out_vid, torch::Tensor counts,
    const torch::Tensor in_vid,
    const torch::Tensor dists, const torch::Tensor inds,
    int ps, int strideIn, int strideOut, int pt,
    int dilation, bool reflect_bounds, int patch_offset){

  // -- unpack --
  int B = inds.size(0);
  int HD = inds.size(1);
  int Q = inds.size(2);
  int K = inds.size(3);
  int q_per_thread = 2;

  // -- check dims --
  int inH = in_vid.size(4);
  int inW = in_vid.size(5);
  int outH = out_vid.size(4);
  int outW = out_vid.size(5);
  assert(inH <= outH);
  assert(inW <= outW);

  // -- kernel threads --
  int MAX_THREADS = 512;//1024
  int k_threads = 8;
  int q_threads = MAX_THREADS/(k_threads); // num of queries threads per block
  q_threads = min(Q,q_threads);
  int q_blocks = (Q-1)/(q_per_thread*q_threads)+1;
  int k_blocks = (K-1)/(k_threads)+1;
  dim3 nthreads(q_threads,k_threads);

  // -- kernel blocks --
  dim3 nblocks(q_blocks,k_blocks,B*HD);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(in_vid.type(), "scatter_add_forward_kernel", ([&] {
    scatter_add_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        out_vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        counts.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
        in_vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
        ps, strideIn, strideOut, pt, dilation, reflect_bounds, patch_offset,
        q_per_thread);
    }));
}

