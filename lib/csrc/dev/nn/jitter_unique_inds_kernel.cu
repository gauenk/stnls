
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <assert.h>
#include "shared_nn_utils.cu"

__global__ void jitter_unique_inds_kernel(
    torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> inds,
    int K, int tgt_K, int H, int W, int sqrt_K, int sqrt_K2,
    int q_per_thread, int elems_per_thread){

  // -- shared memory --
  // the performance depends on "K" because of bank conflicts for shared memory.
  // if K + int(sqrt(K)+0.99)^2 > 32, I think this will start. Think K > 16.
  // __shared__ int s[256*(16+16)];
  extern __shared__ bool s[];
  int mem_start = threadIdx.x*elems_per_thread;
  bool* repl = (bool*)&s[mem_start]; // size (K,)
  bool* avail = (bool *)&s[mem_start+K]; // size (sqrt_K,sqrt_K)
  // bool repl[16];
  // bool avail[16];

  // -- alloc --
  int qi;
  int Q = inds.size(0);
  bool check;
  int deltas[3];
  int anchors[3];
  int refs[3];
  int a_index;
  int k_shift = sqrt_K2;
  int AMAX = sqrt_K * sqrt_K;

  // -- cuda threads --
  int qi_thread = q_per_thread*(threadIdx.x + blockDim.x * blockIdx.x);

  // -- for each location --
  for (int qi_ix = 0; qi_ix < q_per_thread; qi_ix++){

    // -- current query --
    qi = qi_thread + qi_ix;
    if (qi >= Q){ continue; }

    //
    // -- init --
    //

#pragma unroll
    for(int i = 0; i < sqrt_K; i++){
      for(int j = 0; j < sqrt_K; j++){
        avail[j+sqrt_K*i] = true;
      }
    }
#pragma unroll
    for(int i = 0; i < 3; i++){
      refs[i] = inds[qi][0][i];
      anchors[i] = refs[i];
    }

    // -- shift center of available locations --
    int a_shift;
    a_shift = min(0,anchors[1] - sqrt_K) + max(0,anchors[1] + sqrt_K - (H-1));
    anchors[1] -= a_shift;
    a_shift = min(0,anchors[2] - sqrt_K) + max(0,anchors[2] + sqrt_K - (W-1));
    anchors[2] -= a_shift;

    // -- init "replace" with "no thanks" --
    for(int i = 0; i < K; i++){
      repl[i] = false;
    }

    //
    // -- Fill Available Locations & Duplicate K Values --
    //

    for(int i = 0; i < K; i++){

      // -- mark "not available" if within radius --
      check = true;
      #pragma unroll
      for(int l = 1; l < 3; l++){ // do _not_ check for time.
        deltas[l] = (inds[qi][i][l] - anchors[l]) + sqrt_K2;
	check = check & ((deltas[l] >= 0) && (deltas[l] < sqrt_K));
      }
      if (check){
        a_index = deltas[1] + sqrt_K*deltas[2];
	avail[a_index] = false;
      }

      // -- mark as "duplicate" if equality --
      for(int j = i+1; j < K; j++){
        check = true;
#pragma unroll
        for(int l = 0; l < 3; l++){
          check = check & (inds[qi][i][l] == inds[qi][j][l]);
        }
        if (check){
          repl[j] = true;
        }
      }

    }

    //
    // -- Replaced Marked Neighbors --
    //

    // index in a spiral; [credit: Nicolas @ "stackoverflow.com spiral" #3706219]
    int tmp;
    int delta_i = 0;
    int delta_j = -1;
    int avail_i = 0;
    int avail_j = 0;
    int inc = 0;
    bool avail_b;
    bool legal;
    // a_index = (1 + sqrt_K2) + sqrt_K * (0+sqrt_K2);
    // avail[a_index] = false;
    a_index = sqrt_K2 + sqrt_K * sqrt_K2;
    // avail[a_index] = false;
    // avail[a_index+1] = false;
    // avail[a_index+2] = false;

    for (int i = 0; i < K; i++){
      if (repl[i] == false){continue;}
      
      // -- loop in a sprial until we find an open spot --
      legal = check_interval(anchors[1]+avail_i,0,H);
      legal = legal && check_interval(anchors[2]+avail_j,0,W);
      // a_index = avail_i + sqrt_K2;
      // a_index += sqrt_K*(avail_j + sqrt_K2);
      if (a_index >= AMAX){
	avail_b = true;
      }else{
	avail_b = avail[a_index] && legal;
	avail[a_index] = false;
      }
      while ((avail_b == false) && (a_index < AMAX)){

	// -- update in spiral --
        check = (avail_i == avail_j);
        check = check || ((avail_i < 0) && (avail_i == -avail_j));
        check = check || ((avail_i > 0) && (avail_i == (1-avail_j)));
        if (check){
	  tmp = delta_i;
          delta_i = -delta_j;
          delta_j = tmp;
        }
        avail_i += delta_i;
        avail_j += delta_j;

	// -- get raster index --
        a_index = avail_i + sqrt_K2;
        a_index += sqrt_K*(avail_j + sqrt_K2);
	if (a_index >= AMAX){ break; }

	// -- check bounds --
	legal = check_interval(anchors[1]+avail_i,0,H);
	legal = legal && check_interval(anchors[2]+avail_j,0,W);
	avail_b = avail[a_index] && legal; // read next
	avail[a_index] = false;
      }

      // -- fill --
      // inds[qi][i][0] = a_index;
      // inds[qi][i][1] = anchors[1] + avail_i;
      // inds[qi][i][2] = avail_j;
      // inds[qi][i][1] = anchors[1] + avail_i;// + a_index;
      // inds[qi][i][2] = anchors[2] + avail_j;
      inds[qi][i][1] = bounds(anchors[1] + avail_i,H);
      inds[qi][i][2] = bounds(anchors[2] + avail_j,W);

    }

  }
}


void jitter_unique_inds_cuda(
     torch::Tensor inds,
     int tgt_K, int H, int W){
  
  // -- unpack --
  int Q = inds.size(0);
  int K = inds.size(1);
  int sqrt_K = max(int(std::sqrt(tgt_K)+0.999),3);
  int sqrt_K2 = sqrt_K/2;
  // fprintf(stdout,"K,tgt_K: %d,%d\n",K,tgt_K);
  // fprintf(stdout,"H,W: %d,%d\n",H,W);

  // -- num 2 run --
  int nRun = Q;

  // -- kernel params --
  int q_per_thread = 1;
  int _nthreads = 256;
  dim3 nthreads(_nthreads);
  int _nblocks = (nRun-1)/(_nthreads*q_per_thread)+1;
  dim3 nblocks(_nblocks);
  // fprintf(stdout,"nblocks,nthreads: %d,%d\n",_nblocks,_nthreads);

  // -- shared memory size --
  int elems_per_thread = K + sqrt_K * sqrt_K;
  int SMEM_NUM = _nthreads * elems_per_thread;
  // fprintf(stdout,"SMEM_NUM: %d,%d\n",SMEM_NUM,sizeof(bool));

  // -- launch kernel --
  jitter_unique_inds_kernel<<<nblocks, nthreads, sizeof(bool) * SMEM_NUM>>>(
       inds.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
       K,tgt_K,H,W,sqrt_K,sqrt_K2,q_per_thread,elems_per_thread);

}
