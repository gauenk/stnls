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

template <typename scalar_t, typename itype, bool INTERPOLATE>
__global__ void scatter_add_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> out_vid,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> counts,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> in_vid,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dists,
    const torch::PackedTensorAccessor32<itype,5,torch::RestrictPtrTraits> inds,
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
    int query_start = q_per_thread*(threadIdx.x + blockDim.x*blockIdx.x);
    int ki = blockIdx.y*blockDim.y+threadIdx.y;
    int ihead = blockIdx.z/B;
    int ibatch = (blockIdx.z-ihead*B) % B;
    if (ki >= K){ return; }

    // -- pixel locations --
    int qi;
    bool valid;
    scalar_t pix,weight;
    int nl_ti,ref[3],ref_p[3];
    itype nl[3],nl_p[3];
    int nW = (inW-1)/strideIn+1;
    int nHW = nW*((inH-1)/strideIn+1);

    // -- across queries --
    for(int _qi = 0; _qi < q_per_thread; _qi++){

      // -- query index --
      qi = query_start + _qi;
      if (qi >= Q){ continue; }

      // -- write location --
      get_pixel_loc(ref,qi,strideIn,nW,nHW,inH,inW);

      // -- non-local index --
      get_pixel_loc(nl,qi,strideOut,nW,nHW,outH,outW);
  #pragma unroll
      for (int _idx=0; _idx < 3; _idx++){
        nl[_idx] = nl[_idx] + inds[ibatch][ihead][qi][ki][_idx];
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
        nl_p[0] = nl[0];
        nl_p[1] = nl[1]+dilation*(pi + patch_offset);
        nl_p[1] = reflect_bounds ? bounds(nl_p[1],inH) : nl_p[1];
        nl_p[2] = nl[2]+dilation*(pj + patch_offset);
        nl_p[2] = reflect_bounds ? bounds(nl_p[2],inW) : nl_p[2];
        check_bounds(valid, nl_p, T, inH, inW);
        if (not valid){ continue; }

        // -- iterate over loop --
        for(int pk = 0; pk < pt; pk++){

          // -- time is always valid --
          ref_p[0] = ref[0] + pk;
          nl_p[0] = reflect_bounds ? bounds(nl[0]+pk,T) : (nl[0]+pk);
          nl_ti = nl_p[0];
          valid = (nl_p[0] >= 0) and (nl_p[0] < T) and (ref_p[0] >= 0) and (ref_p[0] < T);
          if (not valid){ continue; }

          // -- channels --
          for(int iftr = 0; iftr < F; iftr++){

            // -- read --
            pix = weight*in_vid[ibatch][ihead][ref_p[0]][iftr][ref_p[1]][ref_p[2]];
            // atomicAdd(&out_vid[ibatch][ihead][nl_ti][iftr][nl[1]][nl[2]],pix);

            // -- write --
            if (INTERPOLATE){
              bilin2d_assign(pix,(scalar_t)nl_p[1],(scalar_t)nl_p[2],inH,inW,
                             out_vid[ibatch][ihead][nl_ti][iftr]);
            }else{
              atomicAdd(&out_vid[ibatch][ihead][nl_p[0]][iftr][nl_p[1]][nl_p[2]],pix);
            }


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
    int dilation, bool reflect_bounds, int patch_offset, bool itype_int){

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
  if (itype_int){
    AT_DISPATCH_FLOATING_TYPES(in_vid.type(),
                               "scatter_add_int_forward_kernel", ([&] {
    scatter_add_forward_kernel<scalar_t,int,true><<<nblocks, nthreads>>>(
          out_vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          counts.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
          in_vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
          ps, strideIn, strideOut, pt, dilation, reflect_bounds, patch_offset,
          q_per_thread);
      }));
  }else{
    AT_DISPATCH_FLOATING_TYPES(in_vid.type(),
                               "scatter_add_bilin2d_forward_kernel", ([&] {
    scatter_add_forward_kernel<scalar_t,scalar_t,true><<<nblocks, nthreads>>>(
          out_vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          counts.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
          in_vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
          dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
          inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
          ps, strideIn, strideOut, pt, dilation, reflect_bounds, patch_offset,
          q_per_thread);
      }));
  }
}



/************************************

  Backward Pass (for Vid & Dists)

*************************************/

template <typename scalar_t>
__global__ void scatter_add_int_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> in_vid_grad,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dists_grad,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> out_vid_grad,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dists,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
    int ps, int strideIn, int strideOut, int pt, int dilation,
    bool reflect_bounds, int patch_offset, int q_per_thread){

  // -- shape --
  int B =  dists.size(0);
  int HD = dists.size(1);
  int Q =  dists.size(2);
  int K =  dists.size(3);
  int T = out_vid_grad.size(2);
  int F = out_vid_grad.size(3);
  int inH = in_vid_grad.size(4);
  int inW = in_vid_grad.size(5);
  int outH = out_vid_grad.size(4);
  int outW = out_vid_grad.size(5);

  // -- pixel indexing --
  int qi;
  int ref[3],ref_p[3],nl[3],nl_p[3];
  bool valid;
  float weight,grad,pix_m;

  // -- batching --
  int query_start = q_per_thread*(threadIdx.x + blockDim.x*blockIdx.x);
  int ki = blockIdx.y*blockDim.y+threadIdx.y;
  int ihead = blockIdx.z/B;
  int ibatch = (blockIdx.z-ihead*B) % B;
  if (ki >= K){ return; }
  int nW = (outW-1)/strideOut+1;
  int nHW = nW*((outH-1)/strideOut+1);

  // -- across queries --
  for(int _qi = 0; _qi < q_per_thread; _qi++){

    // -- query index --
    qi = query_start + _qi;
    if (qi >= Q){ continue; }
    get_pixel_loc<int>(ref,qi,strideOut,nW,nHW,inH,inW);

    // -- non-local index --
#pragma unroll
    get_pixel_loc<int>(nl,qi,strideIn,nW,nHW,outH,outW);
    for (int _idx=0; _idx < 3; _idx++){
      nl[_idx] = nl[_idx] + inds[ibatch][ihead][qi][ki][_idx];
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
        check_bounds(valid, ref_p, T, inH, inW);
        if (not valid){ continue; }
  
        // // -- increment legal refs --
        // if ((ref_p[0]==0) and (ibatch==0) and (ihead==0) and (ki==0)){
        //   atomicAdd(&counts[ref_p[1]][ref_p[2]],1);
        // }
  
        // -- non-local pixel index --
        nl_p[0] = nl[0];
        nl_p[1] = nl[1]+dilation*(pi + patch_offset);
        nl_p[1] = reflect_bounds ? bounds(nl_p[1],outH) : nl_p[1];
        nl_p[2] = nl[2]+dilation*(pj + patch_offset);
        nl_p[2] = reflect_bounds ? bounds(nl_p[2],outW) : nl_p[2];
        check_bounds(valid, nl_p, T, outH, outW);
        if (not valid){ continue; }

        // -- init accumulation --
        scalar_t acc_dists_grad = 0;

        for (int pk = 0; pk < pt; pk++){

          // -- time is always valid --
          ref_p[0] = ref[0] + pk;
          nl_p[0] = reflect_bounds ? bounds(nl[0]+pk,T) : (nl[0]+pk);
          valid = (nl_p[0] >= 0) and (nl_p[0] < T) and (ref_p[0] >= 0) and (ref_p[0] < T);
          if (not valid){ continue; }

          // -- num features --
          for (int iftr = 0; iftr < F; iftr++){
            grad = out_vid_grad[ibatch][ihead][nl_p[0]][iftr][nl_p[1]][nl_p[2]];
            pix_m = vid[ibatch][ihead][ref_p[0]][iftr][ref_p[1]][ref_p[2]];
            atomicAdd(&in_vid_grad[ibatch][ihead][ref_p[0]][iftr][ref_p[1]][ref_p[2]],
                      weight*grad);
            acc_dists_grad += grad*pix_m;
          }

        } // pt

      // -- write dist grad --
      atomicAdd(&dists_grad[ibatch][ihead][qi][ki],acc_dists_grad);

    }} // pi,pj
  } // qi
}

void scatter_add_int_backward_cuda(
    torch::Tensor in_vid_grad, torch::Tensor dists_grad,
    const torch::Tensor out_vid_grad, const torch::Tensor vid,
    const torch::Tensor dists, const torch::Tensor inds,
    int ps, int strideIn, int strideOut, int pt, int dilation,
    bool reflect_bounds, int patch_offset){

  // -- launch parameters --
  int B = dists.size(0);
  int HD = dists.size(1);
  int Q = dists.size(2);
  int K = dists.size(3);
  int q_per_thread = 1;
  
  // -- kernel threads --
  int MAX_THREADS = 512;
  int k_threads = 8;
  int q_threads = MAX_THREADS/(k_threads); // num of queries threads per block
  q_threads = min(Q,q_threads);
  int q_blocks = (Q-1)/(q_per_thread*q_threads)+1;
  int k_blocks = (K-1)/(k_threads)+1;
  dim3 nthreads(q_threads,k_threads);

  // -- kernel blocks --
  dim3 nblocks(q_blocks,k_blocks,B*HD);

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(in_vid_grad.type(),
                             "scatter_add_int_backward_vid_kernel", ([&] {
    scatter_add_int_backward_kernel<scalar_t><<<nblocks, nthreads>>>(
        in_vid_grad.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        dists_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        out_vid_grad.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
        ps, strideIn, strideOut, pt, dilation, reflect_bounds, patch_offset,
        q_per_thread);
      }));
  
}

/************************************

  Bilin2d Backward Pass (for Vid & Dists)

*************************************/

template <typename scalar_t>
__global__ void scatter_add_bilin2d_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> in_vid_grad,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dists_grad,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> inds_grad,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> out_vid_grad,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dists,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> inds,
    int ps, int strideIn, int strideOut, int pt, int dilation,
    bool reflect_bounds, int patch_offset, int q_per_thread){


  // -- shape --
  int B =  dists.size(0);
  int HD = dists.size(1);
  int Q =  dists.size(2);
  int K =  dists.size(3);
  int T = out_vid_grad.size(2);
  int F = out_vid_grad.size(3);
  int inH = in_vid_grad.size(4);
  int inW = in_vid_grad.size(5);
  int outH = out_vid_grad.size(4);
  int outW = out_vid_grad.size(5);

  // -- pixel indexing --
  bool valid;
  int qi;
  scalar_t weight,grad,pix_m;
  int ref[3],ref_p[3],nl_ti;
  scalar_t nl[3],nl_p[3];

  // -- batching --
  int query_start = q_per_thread*(threadIdx.x + blockDim.x*blockIdx.x);
  int ki = blockIdx.y*blockDim.y+threadIdx.y;
  int ihead = blockIdx.z/B;
  int ibatch = (blockIdx.z-ihead*B) % B;
  if (ki >= K){ return; }
  int nW = (inW-1)/strideIn+1;
  int nHW = nW*((inH-1)/strideIn+1);

  // -- across queries --
  for(int _qi = 0; _qi < q_per_thread; _qi++){

    // -- query index --
    qi = query_start + _qi;
    if (qi >= Q){ continue; }
    get_pixel_loc(ref,qi,strideOut,nW,nHW,inH,inW);

    // -- non-local index --
#pragma unroll
    get_pixel_loc(nl,qi,strideIn,nW,nHW,outH,outW);
    for (int _idx=0; _idx < 3; _idx++){
      nl[_idx] = nl[_idx] + inds[ibatch][ihead][qi][ki][_idx];
    }

    // -- check "inf" (but it won't be inf sometimes)  --
    valid = (abs(nl[1]) < 1e7) and (abs(nl[2]) < 1e7);
    if (not(valid)){ continue; }

    // -- always reflect anchor point --
    int signH0,signW0;
    nl[0] = bounds(nl[0],T);
    signH0 = check_bound(nl[1],outH) ? 1 : -1;
    nl[1] = bounds(nl[1],outH);
    signW0 = check_bound(nl[2],outW) ? 1 : -1;
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
        check_bounds(valid, ref_p, T, inH, inW);
        if (not valid){ continue; }
  
        // // -- increment legal refs --
        // if ((ref_p[0]==0) and (ibatch==0) and (ihead==0) and (ki==0)){
        //   atomicAdd(&counts[ref_p[1]][ref_p[2]],1);
        // }
  
        // -- non-local pixel index --
        nl_p[0] = nl[0];
        nl_p[1] = nl[1]+dilation*(pi + patch_offset);
        int signH = check_bound(nl_p[1],outH) ? signH0 : -signH0;
        nl_p[1] = reflect_bounds ? bounds(nl_p[1],outH) : nl_p[1];
        nl_p[2] = nl[2]+dilation*(pj + patch_offset);
        int signW = check_bound(nl_p[2],outW) ? signW0 : -signW0;
        nl_p[2] = reflect_bounds ? bounds(nl_p[2],outW) : nl_p[2];
        check_bounds(valid, nl_p, T, outH, outW);
        if (not valid){ continue; }

        // -- gradient accumulation --
        scalar_t acc_dists_grad = 0;
        scalar_t acc_igradH = 0;
        scalar_t acc_igradW = 0;
        scalar_t igradH = 0;
        scalar_t igradW = 0;

        // -- time patch --
        for (int pk = 0; pk < pt; pk++){

          // -- time is always valid --
          ref_p[0] = ref_p[0] + pk;
          nl_ti = reflect_bounds ? bounds(nl[0]+pk,T) : (nl[0]+pk);
          valid = (nl_p[0] >= 0) and (nl_p[0] < T);
          valid = valid and (ref_p[0] >= 0) and (ref_p[0] < T);
          if (not valid){ continue; }

          // -- num features --
          for (int iftr = 0; iftr < F; iftr++){

            // -- read gradient --
            // grad = out_vid_grad[ibatch][ihead][ref_p[0]][iftr][ref_p[1]][ref_p[2]];
            // bilin2d_interpolate(grad,(scalar_t)nl_p[1],(scalar_t)nl_p[2],inH,inW,
            //                     out_vid_grad[ibatch][ihead][nl_ti][iftr]);

            // bilin2d_read_bwd(igradW, igradH, pix_m, grad,
            //                  nl_p[1], nl_p[2], outH, outW,
            //                  out_vid_grad[ibatch][ihead][nl_ti][iftr]);
            pix_m = vid[ibatch][ihead][ref_p[0]][iftr][ref_p[1]][ref_p[2]];
            // bilin2d_assign_bwd(igradW, igradH, pix_m,
            //                    weight*grad, nl_p[1], nl_p[2], inH, inW,
            //                    vid[ibatch][ihead][nl_ti][iftr],
            //                    in_vid_grad[ibatch][ihead][nl_ti][iftr]);


            // grad = out_vid_grad[ibatch][ihead][nl_p[0]][iftr][nl_p[1]][nl_p[2]];
            // pix_m = vid[ibatch][ihead][ref_p[0]][iftr][ref_p[1]][ref_p[2]];
            // atomicAdd(&in_vid_grad[ibatch][ihead][ref_p[0]][iftr][ref_p[1]][ref_p[2]],
            //           weight*grad);
            // acc_dists_grad += grad*pix_m;

            // -- write at in_vid_grad
            atomicAdd(&in_vid_grad[ibatch][ihead][ref_p[0]][iftr][ref_p[1]][ref_p[2]],
                      weight*pix_m);

            // -- accumulate dists --
            acc_dists_grad += grad*pix_m;
            acc_igradW += grad*igradW;
            acc_igradH += grad*igradH;
          }

        } // pt

      // -- write dist grad --
      atomicAdd(&dists_grad[ibatch][ihead][qi][ki],acc_dists_grad);

      // -- write flows grad --
      atomicAdd(&inds_grad[ibatch][ihead][qi][ki][1],weight*acc_igradH*signH);
      atomicAdd(&inds_grad[ibatch][ihead][qi][ki][2],weight*acc_igradW*signW);

    }} // pi,pj
  } // qi

}

void scatter_add_bilin2d_backward_cuda(
    torch::Tensor in_vid_grad,
    torch::Tensor dists_grad, torch::Tensor inds_grad,
    const torch::Tensor out_vid_grad, const torch::Tensor vid,
    const torch::Tensor dists, const torch::Tensor inds,
    int ps, int strideIn, int strideOut, int pt, int dilation,
    bool reflect_bounds, int patch_offset){

  // -- launch parameters --
  int B = dists.size(0);
  int HD = dists.size(1);
  int Q = dists.size(2);
  int K = dists.size(3);
  int q_per_thread = 2;
  
  // -- kernel threads --
  int MAX_THREADS = 512;
  int k_threads = 8;
  int q_threads = MAX_THREADS/(k_threads); // num of queries threads per block
  q_threads = min(Q,q_threads);
  int q_blocks = (Q-1)/(q_per_thread*q_threads)+1;
  int k_blocks = (K-1)/(k_threads)+1;
  dim3 nthreads(q_threads,k_threads);

  // -- kernel blocks --
  dim3 nblocks(q_blocks,k_blocks,B*HD);

  // launch kernel
  AT_DISPATCH_FLOATING_TYPES(in_vid_grad.type(),
                             "scatter_add_bilin2d_backward_vid_kernel", ([&] {
    scatter_add_bilin2d_backward_kernel<scalar_t><<<nblocks, nthreads>>>(
        in_vid_grad.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        dists_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        inds_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        out_vid_grad.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        ps, strideIn, strideOut, pt, dilation, reflect_bounds, patch_offset,
        q_per_thread);
      }));
  
}
