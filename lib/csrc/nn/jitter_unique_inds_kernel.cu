
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <assert.h>

__global__ void jitter_unique_inds_kernel(
    torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> inds,
    int K, int tgt_K, int H, int W, int sqrt_K, int sqrt_K2,
    int q_per_thread, int mem_per_thread){

  // -- shared memory --
  extern __shared__ bool s[];
  int mem_start = threadIdx.x*mem_per_thread;
  bool* repl = (bool*)&s[mem_start]; // size (K,)
  bool* avail = (bool *)(&s[mem_start+K]); // size (sqrt_K,sqrt_K)
  // bool repl[10];
  // bool avail[100];

  // -- alloc --
  int qi;
  int Q = inds.size(0);
  bool check;
  int deltas[3];
  int refs[3];
  int a_index;

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
        avail[i+sqrt_K*j] = (i != sqrt_K2) && (j != sqrt_K2);
      }
    }
#pragma unroll
    for(int i = 0; i < 3; i++){
      refs[i] = inds[qi][0][i+1];
    }
    for(int i = 0; i < K; i++){
      repl[i] = 0;
    }

    //
    // -- Fill Available Locations & Duplicate K Values --
    //

    for(int i = 0; i < K; i++){

      // -- mark "not available" if within radius --
      if(i > 0){
        check = true;
	#pragma unroll
        for(int l = 0; l < 3; l++){
          deltas[l] = refs[l] - inds[qi][i][l];
	  int lim = (l==0) ? 1e-10  : sqrt_K2;
	  check = check & (abs(deltas[l]) < lim);
        }
        if (check){
          a_index = deltas[0]+sqrt_K2;
          a_index += sqrt_K*(deltas[1]+sqrt_K2);
          avail[a_index] = 0;
        }
      }

      // -- mark as "duplicate" if equality --
      for(int j = i+1; j < K; j++){
        check = true;
#pragma unroll
        for(int l = 0; l < 3; l++){
          check = check & (inds[qi][i][l] == inds[qi][j][l]);
        }
        if (check){
          repl[j] = 1;
        }
      }

    }

    //
    // -- Replaced Marked Neighbors --
    //

    // index in a spiral; [credit: stackoverflow.com #3706219]

    int delta_i = 0;
    int delta_j = -1;
    int avail_i = 0;
    int avail_j = 0;
    a_index = sqrt_K2 + sqrt_K * sqrt_K2;

    for (int i = 0; i < K; i++){
      if (repl[i] == 0){continue;}
      
      // -- loop in a sprial until we find an open spot --
      while (avail[a_index] == 0){
        check = (avail_i == avail_j);
        check = check || ((avail_i < 0) && (avail_i == -avail_j));
        check = check || ((avail_i > 0) && (avail_i == (1-avail_j)));
        if (check){
          delta_i = -delta_j;
          delta_j = delta_j;
        }
        avail_i += delta_i;
        avail_j += delta_j;
        a_index = avail_i + sqrt_K2;
        a_index += sqrt_K*(avail_j + sqrt_K2);
      }

      // -- fill --
      inds[qi][i][1] = refs[0] + avail_i;
      inds[qi][i][2] = refs[1] + avail_j;

    }

  }
}


void jitter_unique_inds_cuda(
     torch::Tensor inds,
     int tgt_K, int H, int W){
  
  // -- unpack --
  int Q = inds.size(0);
  int K = inds.size(1);
  int sqrt_K = std::sqrt(tgt_K);
  int sqrt_K2 = sqrt_K/2;

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
  int mem_per_thread = K * sqrt_K * sqrt_K;
  int SMEM = _nthreads * mem_per_thread;

  // -- launch kernel --
  jitter_unique_inds_kernel<<<nblocks, nthreads, SMEM>>>(
       inds.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
       K,tgt_K,H,W,sqrt_K,sqrt_K2,q_per_thread,mem_per_thread);

}
