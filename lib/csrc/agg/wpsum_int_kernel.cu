
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
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> in_vid,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> out_vid,
    torch::PackedTensorAccessor32<int,6,torch::RestrictPtrTraits> counts,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dists,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
    int ps, int pt, int dilation, bool reflect_bounds, int patch_offset,
    int q_per_thread, int f_per_thread){

    // -- shapes --
    int B = in_vid.size(0);
    int HD = in_vid.size(1);
    int T = in_vid.size(2);
    int F = in_vid.size(3);
    int H = in_vid.size(4);
    int W = in_vid.size(5);
    int Q = inds.size(2);
    int k = inds.size(3);

    // -- batching --
    int query_start = blockIdx.x*q_per_thread;
    int ibatch = blockIdx.y;
    int ihead = blockIdx.z;

    // -- cuda threads --
    int pi = threadIdx.y;
    int pj = threadIdx.z;
    int ftr_start = threadIdx.x*f_per_thread;
    int ftr_end = min(F,ftr_start + f_per_thread);

    // -- pixel locations --
    int qi,ti,hi,wi;
    bool valid;
    scalar_t pix,weight;
    int ref_ti;
    int ref[3];
    int nW = (W-1)/stride0+1;
    int nHW = (H-1)/stride0+1;

    // -- range --
    for(int _qi = 0; _qi < q_per_thread; _qi++){

      // -- query index --
      qi = query_start + _qi;
      if (qi >= Q){ continue; }
      get_pixel_loc<int>(ref,qi,stride0,nW,nHW,H,W);

      // -- reference pixel index --
      ref[1] = ref[1]+dilation*(pi + patch_offset);
      ref[2] = ref[2]+dilation*(pj + patch_offset);
      ref[1] = reflect_bounds ? bounds(ref[1],H) : ref[1];
      ref[2] = reflect_bounds ? bounds(ref[2],W) : ref[2];

      for(int ki = 0; ki < k; ki++){

        // -- non-local patch center --
        prop_ti = inds[ibatch][ihead][qi][ki][0];
        prop_hi = inds[ibatch][ihead][qi][ki][1];
        prop_wi = inds[ibatch][ihead][qi][ki][2];
        weight = dists[ibatch][ihead][qi][ki];

        // -- non-local pixel index --
        hi = prop_hi+dilation*(pi + patch_offset);
        wi = prop_wi+dilation*(pj + patch_offset);
        hi = reflect_bounds ? bounds(hi,H) : hi;
        wi = reflect_bounds ? bounds(wi,W) : wi;

        // -- valid non-local patches only --
        valid = (hi >= 0) && (hi < H);
        valid = valid && (wi >= 0) && (wi < W);
        if (not valid){ continue; }

        // -- iterate over loop --
        for(int pk = 0; pk < pt; pk++){

          // -- time is always valid --
          ref_ti = bounds(ref_t + pk,T);
          ti = bounds(prop_ti + pk,T);

          // -- channels --
          for(int iftr = ftr_start; iftr < ftr_end; iftr++){

            // -- fill --
            pix = weight*in_vid[ibatch][ihead][ti][iftr][hi][wi];
            atomicAdd(&out_vid[ibatch][ihead][ref_ti][iftr][ref[1]][ref[2]],pix);

          } // channel-loop
        } // pt-loop
      } // k-loop

      // -- normalize --
      if ((pi==0) and (pj==0) and (ftr_start == 0)){
        if (valid_ref){
          atomicAdd(&counts[ibatch][ihead][0][0][ref[1]][ref[2]],1);
        }
      }

    } // query-loop
}

void wpsum_int_forward_cuda(
    torch::Tensor in_vid,
    torch::Tensor out_vid,
    torch::Tensor counts,
    torch::Tensor dists, torch::Tensor inds,
    int ps, int pt, int dilation,
    bool reflect_bounds, int patch_offset){

  // -- kernel blocks --
  int B = inds.size(0);
  int HD = inds.size(1);
  int Q = inds.size(2);
  int q_per_thread = 2;
  int q_nblocks = (Q-1)/q_per_thread+1;
  dim3 nblocks(q_nblocks,B,HD);

  // -- kernel threads --
  int nftrs = in_vid.size(3);
  int MAX_THREADS = 1024;
  int f_per_block = MAX_THREADS/(ps*ps); // num of nftrs per block
  int f_per_thread = ((nftrs - 1)/f_per_block) + 1; // num of nftrs per thread
  dim3 nthreads(f_per_block,ps,ps);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(in_vid.type(), "wpsum_int_forward_kernel", ([&] {
    wpsum_int_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        in_vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        out_vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        counts.packed_accessor32<int,6,torch::RestrictPtrTraits>(),
        dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
        ps, pt, dilation, reflect_bounds, patch_offset,
        q_per_thread, f_per_thread);
    }));
}



/********************************

     Backward Pass (for Vid)

********************************/

template <typename scalar_t>
__global__ void wpsum_int_backward_vid_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> out_grad,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> in_grad,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dists,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
    int ps, int pt, int dilation, bool reflect_bounds, int patch_offset, int fpt){
    // int qpt, int hpb, int cpt){

  // -- shape --
  int B = dists.size(0);
  int HD = dists.size(1);
  int Q =    dists.size(2);
  int k =     dists.size(3);
  int T = out_grad.size(2);
  int F = out_grad.size(3);
  int H = out_grad.size(4);
  int W = out_grad.size(5);

  // -- pixel indexing --
  int ti,hi,wi;
  int prop_ti,prop_hi,prop_wi;
  bool valid_h,valid_w,valid;
  int ref_t,ref_h,ref_w;
  int ref_ti,ref_hi,ref_w;
  bool valid_ref_h,valid_ref_w,valid_ref;
  float weight,pix;

  // -- location to fill --
  int qi = blockIdx.x*blockDim.x+threadIdx.x;
  int ki = blockIdx.y*blockDim.y+threadIdx.y;
  int ihead = blockIdx.z/B;
  int ibatch = (blockIdx.z-ihead*B) % B;

  // -- feature chunk --
  int ftr_start = threadIdx.z * fpt;
  int ftr_end = min(F,ftr_start + fpt);

  // -- fill --
  if ((qi < Q) && (ki < k)) { // -- if valid --

    // -- reference --
    ref_t = inds[ibatch][ihead][qi][0][0];
    ref_h = inds[ibatch][ihead][qi][0][1];
    ref_w = inds[ibatch][ihead][qi][0][2];

    // -- non-local --
    prop_ti = inds[ibatch][ihead][qi][ki][0];
    prop_hi = inds[ibatch][ihead][qi][ki][1];
    prop_wi = inds[ibatch][ihead][qi][ki][2];
    weight = dists[ibatch][ihead][qi][ki];

    for (int pk = 0; pk < pt; pk++){
      ti = prop_ti + pk;
      ref_ti = ref_t + pk;
    
      for (int pi = 0; pi < ps; pi++){
    
        hi = prop_hi + dilation*(pi + psOffset);
        hi = reflect_bounds ? bounds(hi,H) : hi;
        valid_h = (hi >= 0) && (hi < H);
    
        ref_h = ref_h + dilation*(pi + psOffset);
        ref_h = reflect_bounds ? bounds(ref_h,H) : ref_h;
        valid_ref_h = (ref_h >= 0) && (ref_h < H);
    
        for (int pj = 0; pj < ps; pj++){
    
          wi = prop_wi + dilation*(pj + psOffset);
          wi = reflect_bounds ? bounds(wi,W) : wi;
          valid_w = (wi >= 0) && (wi < W);
    
          ref_w = ref_w + dilation*(pj + psOffset);
          ref_w = reflect_bounds ? bounds(ref_w,W) : ref_w;
          valid_ref_w = (ref_w >= 0) && (ref_w < W);
    
          valid = valid_h && valid_w;
          valid_ref = valid_ref_h && valid_ref_w;
    
          // -- skip if invalid --
          if (not (valid && valid_ref)){ continue; }

          // -- color channels --
          for (int iftr = ftr_start; iftr < ftr_end; iftr++){
            pix = weight * in_grad[ibatch][ihead][ref_ti][iftr][ref_h][ref_w];
            atomicAdd(&out_grad[ibatch][ihead][ti][iftr][hi][wi],pix);
          }
        }
      }
    }
  }
}

void wpsum_int_backward_vid_cuda(
    torch::Tensor out_grad, torch::Tensor in_grad, 
    torch::Tensor dists, torch::Tensor inds,
    int ps, int pt, int dilation,
    bool reflect_bounds, int patch_offset){

  // -- launch parameters --
  int B = dists.size(0);
  int HD = dists.size(1);
  int Q = dists.size(2);
  int k = dists.size(3);
  int nftrs = in_grad.size(3);
  int ftr_threads = min(16,nftrs);
  dim3 threadsPerBlock(16,4,ftr_threads);
  dim3 blocksPerGrid(1, 1, HD*B);
  blocksPerGrid.x = ceil(double(Q)/double(threadsPerBlock.x));
  blocksPerGrid.y = ceil(double(k)/double(threadsPerBlock.y));
  int fpt = (nftrs-1)/ftr_threads+1;

  // -- derivative quantites --
  assert(pt == 1);

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(in_grad.type(), "wpsum_int_backward_vid_kernel", ([&] {
    wpsum_int_backward_vid_kernel<scalar_t><<<blocksPerGrid, threadsPerBlock>>>(
        out_grad.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        in_grad.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
        ps, pt, dilation, reflect_bounds, patch_offset, fpt);
      }));
  
}

/********************************

    Backward Pass (for Dists)

********************************/


template <typename scalar_t>
__global__ void wpsum_int_backward_dists_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dists_grad,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> in_grad,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
    int ps, int pt, int dilation, bool reflect_bounds, int patch_offset, int fpt){

  // -- shapes --
  int B = dists_grad.size(0);
  int Q = dists_grad.size(2);
  int k = dists_grad.size(3);
  int F = in_grad.size(3);
  int H = vid.size(4);
  int W = vid.size(5);

  // -- init registers --
  int ti,hi,wi;
  int ref_ti,ref_hi,ref_w;
  float pix_n,pix_m,pix;
  bool valid_h,valid_w,valid;
  bool valid_ref_h,valid_ref_w,valid_ref;

  // -- location to fill --
  int qi = blockIdx.x*blockDim.x+threadIdx.x;
  int ki = blockIdx.y*blockDim.y+threadIdx.y;
  int ihead = blockIdx.z/B;
  int ibatch = (blockIdx.z-ihead*B) % B;

  // -- feature chunk --
  int ftr_start = threadIdx.z * fpt;
  int ftr_end = min(F,ftr_start + fpt);

  if ((qi < Q) && (ki < k)) { // -- if valid --

    // -- reference --
    int ref_t = inds[ibatch][ihead][qi][0][0];
    int ref_h = inds[ibatch][ihead][qi][0][1];
    int ref_w = inds[ibatch][ihead][qi][0][2];

    // -- non-local --
    int prop_ti = inds[ibatch][ihead][qi][ki][0];
    int prop_hi = inds[ibatch][ihead][qi][ki][1];
    int prop_wi = inds[ibatch][ihead][qi][ki][2];

    for (int pk = 0; pk < pt; pk++){
      ti = prop_ti + pk;
      ref_ti = ref_t + pk;

      for (int pi = 0; pi < ps; pi++){

        hi = prop_hi + dilation*(pi + patch_offset);
        hi = reflect_bounds ? bounds(hi,H) : hi;
        valid_h = (hi >= 0) && (hi < H);

        ref_h = ref_h + dilation*(pi + patch_offset);
        ref_h = reflect_bounds ? bounds(ref_h,H) : ref_h;
        valid_ref_h = (ref_h >= 0) && (ref_h < H);

        for (int pj = 0; pj < ps; pj++){

          wi = prop_wi + dilation*(pj + patch_offset);
          wi = reflect_bounds ? bounds(wi,W) : wi;
          valid_w = (wi >= 0) && (wi < W);

          ref_w = ref_w + dilation*(pj + patch_offset);
          ref_w = reflect_bounds ? bounds(ref_w,W) : ref_w;
          valid_ref_w = (ref_w >= 0) && (ref_w < W);

          valid = valid_h && valid_w;
          valid_ref = valid_ref_h && valid_ref_w;

          // -- skip if invalid --
          if (not (valid && valid_ref)){ continue; }

          for (int iftr = ftr_start; iftr < ftr_end; iftr++){
              pix_n = in_grad[ibatch][ihead][ref_ti][iftr][ref_h][ref_w];
              pix_m = vid[ibatch][ihead][ti][iftr][hi][wi];
              pix = pix_n * pix_m;
              atomicAdd(&dists_grad[ibatch][ihead][qi][ki],pix);
          }
        }
      }
    }
  }
}

void wpsum_int_backward_dists_cuda(
    torch::Tensor dists_grad, torch::Tensor in_grad,
    torch::Tensor vid, torch::Tensor inds,
    int ps, int pt, int dilation, bool reflect_bounds, int patch_offset){

  // -- launch parameters --
  int B = dists_grad.size(0);
  int HD = dists_grad.size(1);
  int Q = dists_grad.size(2);
  int k = dists_grad.size(3);
  int nftrs = vid.size(3);
  int ftr_threads = min(16,nftrs);
  dim3 threadsPerBlock(16,4,ftr_threads);
  dim3 blocksPerGrid(1, 1, HD*B);
  blocksPerGrid.x = ceil(double(Q)/double(threadsPerBlock.x));
  blocksPerGrid.y = ceil(double(k)/double(threadsPerBlock.y));
  int fpt = (nftrs-1)/ftr_threads+1;

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(vid.type(), "wpsum_int_backward_dists_kernel", ([&] {
    wpsum_int_backward_dists_kernel<scalar_t><<<blocksPerGrid, threadsPerBlock>>>(
        dists_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        in_grad.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
        ps, pt, dilation, reflect_bounds, patch_offset, fpt);
  }));
    
}


