/*

  Stack non-local patches into a video

*/

#include "scatter_int.cu"



/**************************************

          Forward Pass

**************************************/

template <typename scalar_t>
__global__ void scatter_int_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weights,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
    const torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> labels,
    torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> stack,
    torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> mask,
    torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> counts,
    int ps, int pt, int dilation, int stride0, bool reflect_bounds,
    int patch_offset, int nW0, int nHW0, int ftrs_per_thread){


    // -- unpack --
    int B = vid.size(0);
    int HD = vid.size(1);
    int T = vid.size(2);
    int F = vid.size(3);
    int H = vid.size(4);
    int W = vid.size(5);
    int Q = inds.size(2);
    int K = inds.size(3);

    // -- indexing variables --
    int iftr;
    int ref_patch[3];
    int nl_patch[3];
    int ref[3];
    int nl[3];
    bool valid;
    bool valid_ref[4];
    bool valid_nl[4];
  
    // -- location to fill --
    int qi = blockIdx.x*blockDim.x+threadIdx.x;
    int ki = blockIdx.y*blockDim.y+threadIdx.y;
    int ihead = blockIdx.z/B;
    int ibatch = (blockIdx.z-ihead*B) % B;
  
    // -- feature chunk --
    int ftr_start = 0;//threadIdx.z * ftrs_per_thread;
    int ftr_end = F;//min(F,ftr_start + ftrs_per_thread);
    
    // -- each region --
    if ((qi < Q) && (ki < K)){
  
      //----------------------------------
      //   Reference & Non-Local Pixel
      //----------------------------------
  
      // -- reference index --
      get_pixel_loc(ref_patch,qi,stride0,nW0,nHW0,H,W);

      // -- non-local index --
  #pragma unroll
      for (int _idx=0; _idx < 3; _idx++){
        nl_patch[_idx] = ref_patch[_idx] + inds[ibatch][ihead][qi][ki][_idx];
      }
      nl_patch[0] = bounds(nl_patch[0],T);
      nl_patch[1] = bounds(nl_patch[1],H);
      nl_patch[2] = bounds(nl_patch[2],W);

      //----------------------------------
      //      Read Scattering Label
      //----------------------------------
      int li = labels[ibatch][ihead][qi][ki];
  
      //----------------------------------
      //      Fill Non-Local Patch
      //----------------------------------
      scatter_patch_fwd_int<scalar_t>(stack[ibatch][ihead][li],
                                      mask[ibatch][ihead][li],
                                      counts[ibatch][ihead],
                                      vid[ibatch][ihead],
                                      weights[ibatch][ihead][qi][ki],
                                      ps,pt,dilation,reflect_bounds,
                                      ref_patch,nl_patch,ref,nl,
                                      valid_ref,valid_nl,valid,
                                      patch_offset,iftr,ftr_start,ftr_end,
                                      T,H,W,qi,ki);

    }

}

void scatter_int_forward_cuda(
    const torch::Tensor vid, const torch::Tensor weights,
    const torch::Tensor inds, const torch::Tensor labels,
    torch::Tensor stack, torch::Tensor mask, torch::Tensor counts,
    int ps, int pt, int dilation, int stride0,
    bool reflect_bounds, int patch_offset){

  // -- sizes --
  int B = vid.size(0);
  int HD = vid.size(1);
  int T = vid.size(2);
  int F = vid.size(3);
  int H = vid.size(4);
  int W = vid.size(5);

  // -- indexing vars --
  int nH0 = (H-1)/stride0+1;
  int nW0 = (W-1)/stride0+1;
  int nHW0 = nH0*nW0;

  // -- launch parameters --
  int Q = inds.size(2);
  int K = inds.size(3);
  int ftr_threads = min(1,F);
  dim3 threadsPerBlock(156,4,ftr_threads);
  dim3 blocksPerGrid(1, 1, HD*B);
  blocksPerGrid.x = ceil(double(Q)/double(threadsPerBlock.x));
  blocksPerGrid.y = ceil(double(K)/double(threadsPerBlock.y));
  int ftrs_per_thread = (F-1)/ftr_threads+1;

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(vid.type(), "scatter_int_forward_kernel", ([&] {
        scatter_int_forward_kernel<scalar_t><<<blocksPerGrid, threadsPerBlock>>>(
           vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
           weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
           inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
           labels.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
           stack.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
           mask.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
           counts.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
           ps, pt, dilation, stride0, reflect_bounds, patch_offset,
           nW0, nHW0, ftrs_per_thread);
      }));

}



// /**************************************

//           Backward Pass

// **************************************/

// template <typename scalar_t>
// __global__ void scatter_int_backward_kernel(
//     torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> grad_vid,
//     torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_weights,
//     const torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> grad_stack,
//     const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid,
//     const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weights,
//     const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
//     // const torch::PackedTensorAccessor32<scalar_t,7,torch::RestrictPtrTraits> stack,
//     const torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> counts,
//     int ps, int pt, int dilation, int stride0, bool reflect_bounds, int patch_offset,
//     int nW0, int nHW0, int ftrs_per_thread){

//     // -- unpack --
//     int B = vid.size(0);
//     int HD = vid.size(1);
//     int T = vid.size(2);
//     int F = vid.size(3);
//     int H = vid.size(4);
//     int W = vid.size(5);
//     int Q = inds.size(2);
//     int K = inds.size(3);

//     // -- indexing variables --
//     int iftr;
//     int ref_patch[3];
//     int nl_patch[3];
//     int ref[3];
//     int nl[3];
//     bool valid;
//     bool valid_ref[4];
//     bool valid_nl[4];
  
//     // -- location to fill --
//     int qi = blockIdx.x*blockDim.x+threadIdx.x;
//     int ki = blockIdx.y*blockDim.y+threadIdx.y;
//     int ihead = blockIdx.z/B;
//     int ibatch = (blockIdx.z-ihead*B) % B;
  
//     // -- feature chunk --
//     int ftr_start = threadIdx.z * ftrs_per_thread;
//     int ftr_end = min(F,ftr_start + ftrs_per_thread);
    
//     // -- each region --
//     if ((qi < Q) && (ki < K)){

  
//       //----------------------------------
//       //   Reference & Non-Local Pixel
//       //----------------------------------
  
//       // -- reference index --
//       get_pixel_loc(ref_patch,qi,stride0,nW0,nHW0,H,W);
  
//       // -- non-local index --
//   #pragma unroll
//       for (int _idx=0; _idx < 3; _idx++){
//         nl_patch[_idx] = ref_patch[_idx]+inds[ibatch][ihead][qi][ki][_idx];
//       }
//       nl_patch[0] = bounds(nl_patch[0],T);
//       nl_patch[1] = bounds(nl_patch[1],H);
//       nl_patch[2] = bounds(nl_patch[2],W);
  
//       //----------------------------------
//       //      Fill Non-Local Patch
//       //----------------------------------

//       scatter_patch_bwd_int<scalar_t>(grad_vid[ibatch][ihead],
//                                       grad_weights[ibatch][ihead],
//                                       counts[ibatch][ihead],
//                                       grad_stack[ibatch][ihead][ki],
//                                       vid[ibatch][ihead],
//                                       weights[ibatch][ihead][qi][ki],
//                                       ps,pt,dilation,reflect_bounds,
//                                       ref_patch,nl_patch,ref,nl,
//                                       valid_ref,valid_nl,valid,
//                                       patch_offset,iftr,ftr_start,ftr_end,
//                                       T,H,W,qi,ki);

//     }

// }

// void scatter_int_backward_cuda(
//     torch::Tensor grad_vid,
//     torch::Tensor grad_weights,
//     const torch::Tensor grad_stack,
//     const torch::Tensor vid,
//     const torch::Tensor weights,
//     const torch::Tensor inds,
//     const torch::Tensor stack,
//     const torch::Tensor counts,
//     int ps, int pt, int dilation, int stride0,
//     bool reflect_bounds, int patch_offset){

//   // -- sizes --
//   int B = vid.size(0);
//   int HD = vid.size(1);
//   int T = vid.size(2);
//   int F = vid.size(3);
//   int H = vid.size(4);
//   int W = vid.size(5);

//   // -- indexing vars --
//   int nW0 = (W-1)/stride0+1;
//   int nH0 = (H-1)/stride0+1;
//   int nHW0 = nH0*nW0;
//   // fprintf(stdout,"ps,pt,stride0: %d,%d,%d\n",ps,pt,stride0);

//   // -- launch parameters --
//   int Q = inds.size(2);
//   int K = inds.size(3);
//   int ftr_threads = min(1,F);
//   dim3 threadsPerBlock(128,4,ftr_threads);
//   dim3 blocksPerGrid(1, 1, HD*B);
//   blocksPerGrid.x = ceil(double(Q)/double(threadsPerBlock.x));
//   blocksPerGrid.y = ceil(double(K)/double(threadsPerBlock.y));
//   int ftrs_per_thread = (F-1)/ftr_threads+1;


//   // -- launch kernel --
//   AT_DISPATCH_FLOATING_TYPES(vid.type(), "scatter_int_backward_kernel", ([&] {
//         scatter_int_backward_kernel<scalar_t><<<blocksPerGrid, threadsPerBlock>>>(
//            grad_vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
//            grad_weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//            grad_stack.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
//            vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
//            weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
//            inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
//            // stack.packed_accessor32<scalar_t,7,torch::RestrictPtrTraits>(),
//            counts.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
//            ps, pt, dilation, stride0, reflect_bounds, patch_offset,
//            nW0, nHW0, ftrs_per_thread);
//       }));


// }
