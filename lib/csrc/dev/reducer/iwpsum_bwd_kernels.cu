
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

/****************************

       Helper Funcs

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

__inline__ int cpu_bounds(int val, int lim ){
  int vval = val;
  if (val < 0){
    vval = -val;
  }else if (val >= lim){
    vval = 2*(lim-1) - val;
  }
  return vval;
}


/********************************

     Backward Pass (for Vid)

********************************/

template <typename scalar_t>
__global__ void iwpsum_backward_vid_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> out_grad,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> in_grad,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dists,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
    int ps, int pt, int dilation,
    bool reflect_bounds, int adj, int fpt){
    // int qpt, int hpb, int cpt){

  // -- shape --
  int nbatch = dists.size(0);
  int nheads = dists.size(1);
  int nq =    dists.size(2);
  int k =     dists.size(3);
  int nframes = out_grad.size(2);
  int nfeatures = out_grad.size(3);
  int height = out_grad.size(4);
  int width = out_grad.size(5);
  int psHalf = ps/2;
  int psOffset = adj - psHalf;

  // -- pixel indexing --
  int ti,hi,wi;
  int center_ti,center_hi,center_wi;
  bool valid_h,valid_w,valid;
  int ref_t,ref_h,ref_w;
  int ref_ti,ref_hi,ref_wi;
  bool valid_ref_h,valid_ref_w,valid_ref;
  float weight,pix;

  // -- location to fill --
  int qi = blockIdx.x*blockDim.x+threadIdx.x;
  int ki = blockIdx.y*blockDim.y+threadIdx.y;
  int ihead = blockIdx.z/nbatch;
  int ibatch = (blockIdx.z-ihead*nbatch) % nbatch;

  // -- feature chunk --
  int ftr_start = threadIdx.z * fpt;
  int ftr_end = min(nfeatures,ftr_start + fpt);

  // -- fill --
  if ((qi < nq) && (ki < k)) { // -- if valid --

    // -- reference --
    ref_t = inds[ibatch][ihead][qi][0][0];
    ref_h = inds[ibatch][ihead][qi][0][1];
    ref_w = inds[ibatch][ihead][qi][0][2];

    // -- non-local --
    center_ti = inds[ibatch][ihead][qi][ki][0];
    center_hi = inds[ibatch][ihead][qi][ki][1];
    center_wi = inds[ibatch][ihead][qi][ki][2];
    weight = dists[ibatch][ihead][qi][ki];

    for (int pk = 0; pk < pt; pk++){
      ti = center_ti + pk;
      ref_ti = ref_t + pk;
    
      for (int pi = 0; pi < ps; pi++){
    
        hi = center_hi + dilation*(pi + psOffset);
        hi = reflect_bounds ? bounds(hi,height) : hi;
        valid_h = (hi >= 0) && (hi < height);
    
        ref_hi = ref_h + dilation*(pi + psOffset);
        ref_hi = reflect_bounds ? bounds(ref_hi,height) : ref_hi;
        valid_ref_h = (ref_hi >= 0) && (ref_hi < height);
    
        for (int pj = 0; pj < ps; pj++){
    
          wi = center_wi + dilation*(pj + psOffset);
          wi = reflect_bounds ? bounds(wi,width) : wi;
          valid_w = (wi >= 0) && (wi < width);
    
          ref_wi = ref_w + dilation*(pj + psOffset);
          ref_wi = reflect_bounds ? bounds(ref_wi,width) : ref_wi;
          valid_ref_w = (ref_wi >= 0) && (ref_wi < width);
    
          valid = valid_h && valid_w;
          valid_ref = valid_ref_h && valid_ref_w;
    
          // -- skip if invalid --
          if (not (valid && valid_ref)){ continue; }

          // -- color channels --
          for (int iftr = ftr_start; iftr < ftr_end; iftr++){
            pix = weight * in_grad[ibatch][ihead][ref_ti][iftr][ref_hi][ref_wi];
            atomicAdd(&out_grad[ibatch][ihead][ti][iftr][hi][wi],pix);
          }
        }
      }
    }
  }
}

void iwpsum_backward_vid_cuda(
    torch::Tensor out_grad, torch::Tensor in_grad, 
    torch::Tensor dists, torch::Tensor inds,
    int ps, int pt, int dilation,
    bool reflect_bounds, bool use_adj){

  // -- launch parameters --
  int nbatch = dists.size(0);
  int nheads = dists.size(1);
  int nq = dists.size(2);
  int k = dists.size(3);
  int nftrs = in_grad.size(3);
  int ftr_threads = min(16,nftrs);
  dim3 threadsPerBlock(16,4,ftr_threads);
  dim3 blocksPerGrid(1, 1, nheads*nbatch);
  blocksPerGrid.x = ceil(double(nq)/double(threadsPerBlock.x));
  blocksPerGrid.y = ceil(double(k)/double(threadsPerBlock.y));
  int fpt = (nftrs-1)/ftr_threads+1;

  // -- derivative quantites --
  int adj = use_adj ? (ps/2) : 0;
  assert(pt == 1);

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(in_grad.type(), "iwpsum_backward_vid_kernel", ([&] {
    iwpsum_backward_vid_kernel<scalar_t><<<blocksPerGrid, threadsPerBlock>>>(
        out_grad.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        in_grad.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
        ps, pt, dilation, reflect_bounds, adj, fpt);
      }));
  
}

/********************************

    Backward Pass (for Dists)

********************************/


template <typename scalar_t>
__global__ void iwpsum_backward_dists_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dists_grad,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> in_grad,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
    int ps, int pt, int dilation, bool reflect_bounds, int psOffset, int fpt){

  // -- shapes --
  int nbatch = dists_grad.size(0);
  int nq = dists_grad.size(2);
  int k = dists_grad.size(3);
  int nfeatures = in_grad.size(3);
  int height = vid.size(4);
  int width = vid.size(5);
  // int psHalf = ps/2;
  // int adj = use_adj ? psHalf : 0;
  // int psOffset = adj - psHalf;

  // -- init registers --
  int ti,hi,wi;
  int ref_ti,ref_hi,ref_wi;
  float pix_n,pix_m,pix;
  bool valid_h,valid_w,valid;
  bool valid_ref_h,valid_ref_w,valid_ref;

  // -- location to fill --
  int qi = blockIdx.x*blockDim.x+threadIdx.x;
  int ki = blockIdx.y*blockDim.y+threadIdx.y;
  int ihead = blockIdx.z/nbatch;
  int ibatch = (blockIdx.z-ihead*nbatch) % nbatch;

  // -- feature chunk --
  int ftr_start = threadIdx.z * fpt;
  int ftr_end = min(nfeatures,ftr_start + fpt);

  if ((qi < nq) && (ki < k)) { // -- if valid --

    // -- reference --
    int ref_t = inds[ibatch][ihead][qi][0][0];
    int ref_h = inds[ibatch][ihead][qi][0][1];
    int ref_w = inds[ibatch][ihead][qi][0][2];

    // -- non-local --
    int center_ti = inds[ibatch][ihead][qi][ki][0];
    int center_hi = inds[ibatch][ihead][qi][ki][1];
    int center_wi = inds[ibatch][ihead][qi][ki][2];

    for (int pk = 0; pk < pt; pk++){
      ti = center_ti + pk;
      ref_ti = ref_t + pk;

      for (int pi = 0; pi < ps; pi++){

        hi = center_hi + dilation*(pi + psOffset);
        hi = reflect_bounds ? bounds(hi,height) : hi;
        valid_h = (hi >= 0) && (hi < height);

        ref_hi = ref_h + dilation*(pi + psOffset);
        ref_hi = reflect_bounds ? bounds(ref_hi,height) : ref_hi;
        valid_ref_h = (ref_hi >= 0) && (ref_hi < height);

        for (int pj = 0; pj < ps; pj++){

          wi = center_wi + dilation*(pj + psOffset);
          wi = reflect_bounds ? bounds(wi,width) : wi;
          valid_w = (wi >= 0) && (wi < width);

          ref_wi = ref_w + dilation*(pj + psOffset);
          ref_wi = reflect_bounds ? bounds(ref_wi,width) : ref_wi;
          valid_ref_w = (ref_wi >= 0) && (ref_wi < width);

          valid = valid_h && valid_w;
          valid_ref = valid_ref_h && valid_ref_w;

          // -- skip if invalid --
          if (not (valid && valid_ref)){ continue; }

          for (int iftr = ftr_start; iftr < ftr_end; iftr++){
              pix_n = in_grad[ibatch][ihead][ref_ti][iftr][ref_hi][ref_wi];
              pix_m = vid[ibatch][ihead][ti][iftr][hi][wi];
              pix = pix_n * pix_m;
              atomicAdd(&dists_grad[ibatch][ihead][qi][ki],pix);
          }
        }
      }
    }
  }
}

void iwpsum_backward_dists_cuda(
    torch::Tensor dists_grad, torch::Tensor in_grad,
    torch::Tensor vid, torch::Tensor inds,
    int ps, int pt, int dilation, bool reflect_bounds, bool use_adj){

  // -- launch parameters --
  int nbatch = dists_grad.size(0);
  int nheads = dists_grad.size(1);
  int nq = dists_grad.size(2);
  int k = dists_grad.size(3);
  int nftrs = vid.size(3);
  int ftr_threads = min(16,nftrs);
  dim3 threadsPerBlock(16,4,ftr_threads);
  dim3 blocksPerGrid(1, 1, nheads*nbatch);
  blocksPerGrid.x = ceil(double(nq)/double(threadsPerBlock.x));
  blocksPerGrid.y = ceil(double(k)/double(threadsPerBlock.y));
  int fpt = (nftrs-1)/ftr_threads+1;

  // -- shared --
  int psHalf = ps/2;
  int adj = use_adj ? psHalf : 0;
  int psOffset = adj - psHalf;

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(vid.type(), "iwpsum_backward_dists_kernel", ([&] {
    iwpsum_backward_dists_kernel<scalar_t><<<blocksPerGrid, threadsPerBlock>>>(
        dists_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        in_grad.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
        ps, pt, dilation, reflect_bounds, psOffset, fpt);
  }));
    
}

