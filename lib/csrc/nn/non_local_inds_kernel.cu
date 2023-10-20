/*************************

Get indices of a non-local search

*****************************/

#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <assert.h>
// #include "../shared_kernel.cu"
#include "shared_flows.cu"


template <typename scalar_t>
__global__ void non_local_inds_kernel(
    torch::PackedTensorAccessor64<int,6,torch::RestrictPtrTraits> inds,
    const torch::PackedTensorAccessor64<scalar_t,5,torch::RestrictPtrTraits> fflow,
    const torch::PackedTensorAccessor64<scalar_t,5,torch::RestrictPtrTraits> bflow,
    int ws, int wt, int nH, int nW, int nHW,
    int stride0, int stride1, bool full_ws,
    int q_per_thread, int ws_per_thread){

  // -- unpack --
  int ibatch = blockIdx.y;
  int raster_index = q_per_thread*blockIdx.x;
  int T = fflow.size(1);
  int H = fflow.size(3);
  int W = fflow.size(4);
  int B = inds.size(0);
  int Q = inds.size(1);
  int St = inds.size(2);
  int Ss_h = inds.size(3);
  int Ss_w = inds.size(4);
  // int wt = (St-1)/2; // St is *always* odd

  // -- temporal search --
  int hj = 0;
  int wj = 0;
  scalar_t hj_acc,wj_acc;

  // -- search space offset --
  int wsHalf = (ws-1)/2;
  int wsOff_h,wsOff_w;
  int ws_i,ws_j;

  // -- reference location --
  int ref[3];
  bool valid_ref;


  // -- get location --
  for (int _qi = 0; _qi < q_per_thread; _qi++){

    // ---------------------------------------
    //          reference location
    // ---------------------------------------

    int qi = raster_index + _qi;
    if (qi >= Q){continue;}

    // -- fill pixel --
    get_pixel_loc(ref,  qi, stride0, nW, nHW, H, W);
    check_bounds(valid_ref,ref,T,H,W);
    if (not(valid_ref)){ continue; }
    // assert((ref[0] >= 0) && (ref[0] < T)); // check "ti"

    // -- shifted time radius --
    int t_shift = min(0,ref[0] - wt) + max(0,ref[0] + wt - (T-1));
    int t_left = max(ref[0] - wt - t_shift,0);
    int t_right = min(T-1,ref[0] + wt - t_shift);

    // ---------------------------------------
    //   spatial radius centered @ offset
    // ---------------------------------------
  
    // -- search region offsets --
    set_search_offsets(wsOff_h,wsOff_w, ref[1], ref[2], stride1,
                       wsHalf, ws, H, W, full_ws);

    // -- search across space --
    hj = ref[1];
    wj = ref[2];
    for (int _xi = 0; _xi < ws_per_thread; _xi++){
      ws_i = threadIdx.x + blockDim.x*_xi;
      if (ws_i >= ws){ continue; }
      for (int _yi = 0; _yi < ws_per_thread; _yi++){
	ws_j = threadIdx.y + blockDim.y*_yi;
	if (ws_j >= ws){ continue; }
  
	// -- set indices --
	inds[ibatch][qi][0][ws_i][ws_j][0] = ref[0];
	inds[ibatch][qi][0][ws_i][ws_j][1] = hj + stride1 * (ws_i - wsOff_h);
	inds[ibatch][qi][0][ws_i][ws_j][2] = wj + stride1 * (ws_j - wsOff_w);
      }
    }

    // ---------------------------------------
    //         forward through time
    // ---------------------------------------

    // -- run right --
    hj_acc = __int2float_rn(ref[1]);
    wj_acc = __int2float_rn(ref[2]);
    int ta = 1;
    int t_prev = ref[0];
    auto flow = fflow;
    for(int tj=ref[0]+1; tj <= t_right; tj++){

      // -- accumulate --
      update_centers_flow_acc(hj_acc,wj_acc,H,W,flow[ibatch][t_prev]);
      hj = bounds(hj_acc,H);
      wj = bounds(wj_acc,W);
      // itype& hj_center, itype& wj_center, int H, int W,
      //   const torch::TensorAccessor<scalar_t,3,torch::RestrictPtrTraits,int32_t> flow)
      // hj_tmp = hj;
      // wj_tmp = wj;
      // hj = int(1.*hj + flow[ibatch][t_prev][1][hj_tmp][wj_tmp] + 0.5);
      // wj = int(1.*wj + flow[ibatch][t_prev][0][hj_tmp][wj_tmp] + 0.5);
      // hj = max(0,min(H-1,hj));
      // wj = max(0,min(W-1,wj));

      // ---------------------------------------
      //   spatial radius centered @ offset
      // ---------------------------------------
  

      // -- search region offsets --
      set_search_offsets(wsOff_h,wsOff_w, hj, wj, stride1,
                         wsHalf, ws, H, W, full_ws);

      // -- search across space --
      for (int _xi = 0; _xi < ws_per_thread; _xi++){
	ws_i = threadIdx.x + blockDim.x*_xi;
	if (ws_i >= ws){ continue; }
	for (int _yi = 0; _yi < ws_per_thread; _yi++){
	  ws_j = threadIdx.y + blockDim.y*_yi;
	  if (ws_j >= ws){ continue; }

	  // -- set indices --
	  inds[ibatch][qi][ta][ws_i][ws_j][0] = tj;
	  inds[ibatch][qi][ta][ws_i][ws_j][1] = hj + stride1 * (ws_i - wsOff_h);
	  inds[ibatch][qi][ta][ws_i][ws_j][2] = wj + stride1 * (ws_j - wsOff_w);
	}
      }

      // -- update previous flow --
      t_prev = tj;

      // -- incriment pre-computed frame index --
      ta++;
    }


    // ---------------------------------------
    //         backwards through time
    // ---------------------------------------

    // -- init --
    hj_acc = __int2float_rn(ref[1]);
    wj_acc = __int2float_rn(ref[2]);
    t_prev = ref[0];
    flow = bflow;
    for(int tj=ref[0]-1; tj >= t_left; tj--){

      // -- accumulate --
      update_centers_flow_acc(hj_acc,wj_acc,H,W,flow[ibatch][t_prev]);
      hj = bounds(hj_acc,H);
      wj = bounds(wj_acc,W);
      // hj_tmp = hj;
      // wj_tmp = wj;
      // hj = int(1.*hj + flow[ibatch][t_prev][1][hj_tmp][wj_tmp] + 0.5);
      // wj = int(1.*wj + flow[ibatch][t_prev][0][hj_tmp][wj_tmp] + 0.5);
      // hj = max(0,min(H-1,hj));
      // wj = max(0,min(W-1,wj));

      // ---------------------------------------
      //   spatial radius centered @ offset
      // ---------------------------------------
  
      // -- search region offsets --
      set_search_offsets(wsOff_h,wsOff_w, hj, wj, stride1,
                         wsHalf, ws, H, W, full_ws);

      // -- search across space --
      for (int _xi = 0; _xi < ws_per_thread; _xi++){
	ws_i = threadIdx.x + blockDim.x*_xi;
	if (ws_i >= ws){ continue; }
	for (int _yi = 0; _yi < ws_per_thread; _yi++){
	  ws_j = threadIdx.y + blockDim.y*_yi;
	  if (ws_j >= ws){ continue; }

	  // -- set indices --
	  inds[ibatch][qi][ta][ws_i][ws_j][0] = tj;
	  inds[ibatch][qi][ta][ws_i][ws_j][1] = hj + stride1 * (ws_i - wsOff_h);
	  inds[ibatch][qi][ta][ws_i][ws_j][2] = wj + stride1 * (ws_j - wsOff_w);
	}
      }

      // -- update previous flow --
      t_prev = tj;

      // -- incriment pre-computed frame index --
      ta++;
    }

    assert(ta == St);//,"Must be equal."

  }
}


void non_local_inds_cuda(
     torch::Tensor inds,
     const torch::Tensor fflow, const torch::Tensor bflow,
     int ws, int wt, int stride0, int stride1, bool full_ws){
  
  // -- unpack --
  int B = inds.size(0);
  int Q = inds.size(1);
  int St = inds.size(2);
  // int Ss_h = inds.size(3);
  // int Ss_w = inds.size(4);
  int H = fflow.size(3);
  int W = fflow.size(4);
  // assert(Ss_h == ws);
  // assert(Ss_w == ws);

  // -- derivative --
  int nH = (H-1)/stride0 + 1;
  int nW = (W-1)/stride0 + 1;
  int nHW = nH*nW;

  // -- one thread per search region --
  int ws_threads = std::min(ws,27);
  // int ws_w_threads = std::min(Ss_w,27);
  int ws_per_thread = ((ws-1)/ws_threads) + 1;
  // int ws_w_per_thread = ((Ss_w-1)/ws_w_threads) + 1;
  dim3 nthreads(ws_threads,ws_threads);

  // -- chunking the blocks --
  int q_per_thread = 2;
  int _nblocks = (Q-1)/(q_per_thread)+1;
  dim3 nblocks(_nblocks,B);

  // -- viz --
  // fprintf(stdout,"nblocks,nthreads: %d,%d\n",_nblocks,_nthreads);
  // fprintf(stdout,"B,Q,_nblocks,nH,nW,stride0: %d,%d,%d,%d,%d,%d\n",
  // 	  B,Q,_nblocks,nH,nW,stride0);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(fflow.type(), "non_local_inds_kernel", ([&] {
     non_local_inds_kernel<scalar_t><<<nblocks, nthreads>>>(
       inds.packed_accessor64<int,6,torch::RestrictPtrTraits>(),
       fflow.packed_accessor64<scalar_t,5,torch::RestrictPtrTraits>(),
       bflow.packed_accessor64<scalar_t,5,torch::RestrictPtrTraits>(),
       ws, wt, nH, nW, nHW, stride0, stride1, full_ws,
       q_per_thread, ws_per_thread);
      }));

}
