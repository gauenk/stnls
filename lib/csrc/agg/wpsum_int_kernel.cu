
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
__global__ void wpsum_int_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> out_vid,
    torch::PackedTensorAccessor32<int,6,torch::RestrictPtrTraits> counts,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> in_vid,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dists,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
    int ps, int stride0, int pt, int dilation, bool reflect_bounds,
    int patch_offset, int q_per_thread){

    // -- shapes --
    int B = in_vid.size(0);
    int HD = in_vid.size(1);
    int T = in_vid.size(2);
    int F = in_vid.size(3);
    int H = in_vid.size(4);
    int W = in_vid.size(5);
    int Q = inds.size(2);
    int K = inds.size(3);

    // -- batching --
    int query_start = (threadIdx.x + blockDim.x*blockIdx.x)*q_per_thread;
    int ibatch = blockIdx.y;
    int ihead = blockIdx.z;

    // -- cuda threads --
    int pi = threadIdx.y;
    int pj = threadIdx.z;

    // -- pixel locations --
    int qi;
    bool valid;
    scalar_t pix,weight;
    int ref_ti,nl_ti;
    int ref[3];
    int nl[3];
    int nW = (W-1)/stride0+1;
    int nHW = nW*((H-1)/stride0+1);

    // -- across queries --
    for(int _qi = 0; _qi < q_per_thread; _qi++){

      // -- query index --
      qi = query_start + _qi;
      if (qi >= Q){ continue; }
      get_pixel_loc<int>(ref,qi,stride0,nW,nHW,H,W);

      // -- reference pixel index --
      ref[1] = ref[1]+dilation*(pi + patch_offset);
      ref[2] = ref[2]+dilation*(pj + patch_offset);

      // -- valid ref pixel only --
      valid = (ref[1] >= 0) && (ref[1] < H);
      valid = valid && (ref[2] >= 0) && (ref[2] < W);
      if (not valid){ continue; }

      for(int ki = 0; ki < K; ki++){

        // -- non-local index --
    #pragma unroll
        for (int _idx=0; _idx < 3; _idx++){
          nl[_idx] = ref[_idx] + inds[ibatch][ihead][qi][ki][_idx];
        }
  
        // -- always reflect anchor point --
        // nl[1] = bounds(nl[1],H);
        // nl[2] = bounds(nl[2],W);

        // -- non-local pixel index --
        nl[1] = nl[1]+dilation*(pi + patch_offset);
        nl[1] = reflect_bounds ? bounds(nl[1],H) : nl[1];
        nl[2] = nl[2]+dilation*(pj + patch_offset);
        nl[2] = reflect_bounds ? bounds(nl[2],W) : nl[2];

        // -- valid non-local patches only --
        valid = (nl[1] >= 0) && (nl[1] < H);
        valid = valid && (nl[2] >= 0) && (nl[2] < W);
        if (not valid){ continue; }

        // -- non-local weight --
        weight = dists[ibatch][ihead][qi][ki];

        // -- iterate over loop --
        for(int pk = 0; pk < pt; pk++){

          // -- time is always valid --
          ref_ti = bounds(ref[0] + pk,T);
          nl_ti = bounds(nl[0] + pk,T);

          // -- channels --
          for(int iftr = 0; iftr < F; iftr++){

            // -- fill --
            pix = weight*in_vid[ibatch][ihead][nl_ti][iftr][nl[1]][nl[2]];
            atomicAdd(&out_vid[ibatch][ihead][ref_ti][iftr][ref[1]][ref[2]],pix);

          } // nfeatures-loop
        } // pt-loop
      } // k-loop

      // -- normalize --
      if ((ref[0]==0) and (ibatch==0) and (ihead==0)){
        atomicAdd(&counts[0][0][0][0][ref[1]][ref[2]],1);
      }
      // atomicAdd(&counts[0][0][0][0][ref[1]][ref[2]],1);

    } // query-loop
}

void wpsum_int_forward_cuda(
    torch::Tensor out_vid, torch::Tensor counts,
    const torch::Tensor in_vid,
    const torch::Tensor dists, const torch::Tensor inds,
    int ps, int stride0, int pt, int dilation,
    bool reflect_bounds, int patch_offset){

  // -- unpack --
  int B = inds.size(0);
  int HD = inds.size(1);
  int Q = inds.size(2);
  int q_per_thread = 2;

  // -- kernel threads --
  int MAX_THREADS = 1024;
  int q_threads = MAX_THREADS/(ps*ps); // num of queries threads per block
  int q_blocks = (Q-1)/(q_per_thread*q_threads)+1;
  dim3 nthreads(q_threads,ps,ps);

  // -- kernel blocks --
  dim3 nblocks(q_blocks,B,HD);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(in_vid.type(), "wpsum_int_forward_kernel", ([&] {
    wpsum_int_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        out_vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        counts.packed_accessor32<int,6,torch::RestrictPtrTraits>(),
        in_vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
        ps, stride0, pt, dilation, reflect_bounds, patch_offset,
        q_per_thread);
    }));
}



/************************************

  Backward Pass (for Vid & Dists)

*************************************/

template <typename scalar_t>
__global__ void wpsum_int_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> in_vid_grad,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dists_grad,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> out_vid_grad,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dists,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
    int ps, int stride0, int pt, int dilation, bool reflect_bounds, int patch_offset,
    int q_per_thread, int k_per_thread){

  // -- shape --
  int B =  dists.size(0);
  int HD = dists.size(1);
  int Q =  dists.size(2);
  int K =  dists.size(3);
  int T = out_vid_grad.size(2);
  int F = out_vid_grad.size(3);
  int H = out_vid_grad.size(4);
  int W = out_vid_grad.size(5);

  // -- pixel indexing --
  int qi,ki;
  int ref[3];
  scalar_t nl[3];
  int ref_ti,nl_ti;
  bool valid;
  float weight,pix_n,pix_m;

  // -- location to fill --
  int q_start = blockIdx.x*blockDim.x+threadIdx.x;
  int k_start = blockIdx.y*blockDim.y+threadIdx.y;
  int ihead = blockIdx.z/B;
  int ibatch = (blockIdx.z-ihead*B) % B;
  int nW = (W-1)/stride0+1;
  int nHW = nW*((H-1)/stride0+1);

  // -- cuda threads --
  int pi = threadIdx.y;
  int pj = threadIdx.z;

  // -- across queries --
  for(int _qi = 0; _qi < q_per_thread; _qi++){

    // -- query index --
    qi = q_start + _qi;
    if (qi >= Q){ continue; }
    get_pixel_loc<int>(ref,qi,stride0,nW,nHW,H,W);

    // -- reference pixel index --
    ref[1] = ref[1]+dilation*(pi + patch_offset);
    ref[2] = ref[2]+dilation*(pj + patch_offset);

    // -- valid ref pixel only --
    valid = (ref[1] >= 0) && (ref[1] < H);
    valid = valid && (ref[2] >= 0) && (ref[2] < W);
    if (not valid){ continue; }

    for(int _ki = 0; _ki < k_per_thread; _ki++){

      // -- non-local index --
      ki = k_start + _ki;
      if (ki >= K){ continue; }
  #pragma unroll
      for (int _idx=0; _idx < 3; _idx++){
        nl[_idx] = ref[_idx] + inds[ibatch][ihead][qi][ki][_idx];
      }

      // -- non-local pixel index --
      nl[1] = nl[1]+dilation*(pi + patch_offset);
      nl[1] = reflect_bounds ? bounds(nl[1],H) : nl[1];
      nl[2] = nl[2]+dilation*(pj + patch_offset);
      nl[2] = reflect_bounds ? bounds(nl[2],W) : nl[2];

      // -- valid non-local patches only --
      valid = (nl[1] >= 0) && (nl[1] < H);
      valid = valid && (nl[2] >= 0) && (nl[2] < W);
      if (not valid){ continue; }

      // -- non-local weight --
      weight = dists[ibatch][ihead][qi][ki];
      scalar_t acc_dists_grad = 0;

      for (int pk = 0; pk < pt; pk++){
        ref_ti = ref[0] + pk;
        nl_ti = nl[0] + pk;
  
        // -- num features --
        for (int iftr = 0; iftr < F; iftr++){
          pix_n = out_vid_grad[ibatch][ihead][ref_ti][iftr][ref[1]][ref[2]];
          pix_m = vid[ibatch][ihead][nl_ti][iftr][nl[1]][nl[2]];
          atomicAdd(&in_vid_grad[ibatch][ihead][nl_ti][iftr][nl[1]][nl[2]],weight*pix_n);
          acc_dists_grad += pix_n*pix_m;
        }

      } // pt

      // -- write dist grad --
      atomicAdd(&dists_grad[ibatch][ihead][qi][ki],acc_dists_grad);

    } // ki
  } // qi
}

void wpsum_int_backward_cuda(
    torch::Tensor in_vid_grad, torch::Tensor dists_grad,
    const torch::Tensor out_vid_grad, const torch::Tensor vid,
    const torch::Tensor dists, const torch::Tensor inds,
    int ps, int stride0, int pt, int dilation, bool reflect_bounds, int patch_offset){

  // -- launch parameters --
  int B = dists.size(0);
  int HD = dists.size(1);
  int Q = dists.size(2);
  int K = dists.size(3);
  int q_per_thread = 2;
  int k_per_thread = K;
  
  // -- kernel threads --
  int MAX_THREADS = 1024;
  int q_threads = MAX_THREADS/(ps*ps); // num of queries threads per block
  int q_blocks = (Q-1)/(q_per_thread*q_threads)+1;
  int k_blocks = (K-1)/k_per_thread+1;
  dim3 nthreads(q_threads,ps,ps);
  dim3 nblocks(q_blocks, k_blocks, HD*B);

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(in_vid_grad.type(), "wpsum_int_backward_vid_kernel", ([&] {
    wpsum_int_backward_kernel<scalar_t><<<nblocks, nthreads>>>(
        in_vid_grad.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        dists_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        out_vid_grad.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
        ps, stride0, pt, dilation, reflect_bounds, patch_offset,
        q_per_thread, k_per_thread);
      }));
  
}

