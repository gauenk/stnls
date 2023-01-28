
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
using namespace at;

/****************************

       Inline Functions

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

inline __host__ __device__
int unravel_index(int& ti, int& hi, int& wi, const int qindex,
                  const int h, const int w, const int hw){
  // index to pixel location
  int i_mod = qindex % hw;
  ti = qindex / hw;
  wi = (i_mod % w);
  hi = (i_mod / w) % h;
}

inline __host__ __device__ int get_backward_window_start(const int index, const int KERNEL_SIZE, const int NEIGHBORHOOD_SIZE)
{
    return (index < KERNEL_SIZE) ? (0) : index - NEIGHBORHOOD_SIZE;
}


/****************************

       Forward Pass

****************************/

template <typename scalar_t>
__global__ void prod_dists_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> dists,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> inds,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> self_dists,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int chnls, int dilation, int stride1,
    bool use_adj, bool reflect_bounds, bool search_abs, bool anchor_self,
    int qpt, int kpt){

  // shapes
  int nbatch = vid0.size(0);
  int nheads = vid0.size(1);
  int nframes = vid0.size(2);
  int color = vid0.size(3);
  int height = vid0.size(4);
  int width = vid0.size(5);
  int n_hw0 = n_h0 * n_w0;
  int nqueries = dists.size(2);
  int k0 = dists.size(3);

  // constants
  float nan = __int_as_float(0xffe00000);
  float inf = __int_as_float(0x7f800000);

  // offsets
  int psHalf = (ps)/2;
  int adj = use_adj ? psHalf : 0;

  // cuda index
  int ibatch = blockIdx.x;
  int head = blockIdx.y;
  int blkDimX = blockDim.y; // num threads in x-block
  int blkDimY = blockDim.z; // num threads in y-block
  int query_start = blockIdx.z*qpt;
  int k_start = threadIdx.x*kpt;
  
  // helpers 
  int qi,ki,qindex;
  int i_mod;

  // head for inds
  int inds_nheads = inds.size(1);
  int ihead = head / inds_nheads;

  // accumulate time offsets
  bool dir_fwd = true; // forward
  bool swap_dir = false;
  int prev_h,prev_w,prev_t;
  int min_t;

  // decls
  int ti,hi,wi;
  int n_ti,n_hi,n_wi;
  int vH,vW,vT,nH,nW,nT;
  bool valid,vvalid,nvalid;
  // bool valid_ti,valid_hi,valid_wi,valid_anchor;
  // bool valid_n_ti,valid_n_hi,valid_n_wi,valid_n;
  bool valid_anchor,valid_n;
  bool vvalid_t,vvalid_h,vvalid_w;
  bool nvalid_t,nvalid_h,nvalid_w;
  bool eq_dim;//eq_ti,eq_hi,eq_wi,
  scalar_t dist,v_pix,n_pix;


  for (int _qidx = 0; _qidx < qpt; _qidx++){

    // -- query start --
    qi = query_start + _qidx;
    if (qi >= nqueries){ continue; }

    // -- extract anchor pixel --
    qindex = qi + qstart;
    i_mod = qindex % n_hw0;
    ti = qindex / n_hw0;
    wi = ((i_mod % n_w0) * stride0) % width ;
    hi = ((i_mod / n_w0) * stride0) % height;

    // -- valid --
    valid_anchor = (ti < nframes) && (ti >= 0);
    valid_anchor = valid_anchor && (hi < height) && (hi >= 0);
    valid_anchor = valid_anchor && (wi < width) && (wi >= 0);

    for (int _kidx = 0; _kidx < kpt; _kidx++){

      // -- block start --
      ki = k_start + _kidx;
      if (ki >= k0){ continue; }
      n_ti = inds[ibatch][ihead][qi][ki][0];
      n_hi = inds[ibatch][ihead][qi][ki][1];
      n_wi = inds[ibatch][ihead][qi][ki][2];
      valid = valid_anchor;
      dist = 0;

      // ---------------------------------
      //
      //  compute delta over patch vol.
      //
      // ---------------------------------
      for (int pk = 0; pk < pt; pk++){
        // -- anchor time --
        vT = bounds(ti + pk,nframes);
        vvalid_t = (vT < nframes) && (vT >= 0);
 
        // -- proposed time --
        nT = bounds(n_ti + pk,nframes);
        nvalid_t = (nT < nframes) && (nT >= 0);
        
        for (int pi = 0; pi < ps; pi++){
          // -- anchor height --
          vH = (hi-h0_off) + dilation*(pi - psHalf + adj);
          vH = reflect_bounds ? bounds(vH,height) : vH;
          vvalid_h = (vH < height) && (vH >= 0);
          
          // -- propose height --
          nH = (n_hi-h1_off) + dilation*(pi - psHalf + adj);
          nH = reflect_bounds ? bounds(nH,height) : nH;
          nvalid_h = (nH < height) && (nH >= 0);
 
 
          for (int pj = 0; pj < ps; pj++){
            
            // -- anchor width --
            vW = (wi-w0_off) + dilation*(pj - psHalf + adj);
            vW = reflect_bounds ? bounds(vW,width) : vW;
            vvalid_w = (vW < width) && (vW >= 0);
 
            // -- propose width --
            nW = (n_wi-w1_off) + dilation*(pj - psHalf + adj);
            nW = reflect_bounds ? bounds(nW,width) : nW;
            nvalid_w = (nW < width) && (nW >= 0);
 
            // -- check valid --
            vvalid = vvalid_t && vvalid_h && vvalid_w;
            nvalid = nvalid_t && nvalid_h && nvalid_w;
 
            // -- all channels --
            for (int ci = 0; ci < chnls; ci++){
 
              // -- get data --
              v_pix = vvalid ? vid0[ibatch][head][vT][ci][vH][vW] : (scalar_t)0.;
              n_pix = nvalid ? vid1[ibatch][head][nT][ci][nH][nW] : (scalar_t)0.;
 
              // -- compute dist --
              dist += v_pix * n_pix;
            }
          }
        }
      }

      // -- dists --
      if (!valid){ dist = nan; }
      dists[ibatch][head][qi][ki] = dist;
 
      // -- final check [put self@index 0] --
      if (anchor_self){
        eq_dim = n_ti == ti;
        eq_dim = eq_dim && (n_hi == hi);
        eq_dim = eq_dim && (n_wi == wi);
        if (eq_dim){
          self_dists[ibatch][head][qi] = dist;
          dists[ibatch][head][qi][ki] = inf;
        }
      }
      
    }
  }
}

void prod_dists_forward_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1,
    torch::Tensor dists, const torch::Tensor inds,
    torch::Tensor self_dists,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int chnls, int dilation, int stride1,
    bool use_adj, bool reflect_bounds, bool search_abs,
    bool anchor_self){

    // # -- launch params --
    // w_threads = min(ws,32)
    // nthreads = (w_threads,w_threads)
    // ws_iters = (ws-1)//w_threads + 1
    // nblocks = (nq-1)//batches_per_block+1

   // fprintf(stdout,"qstart, nqueries: %d,%d\n",qstart,nqueries);
   // launch params
   // our many (too many?) registers limit the number of threads

   // -- comp per threads --
   int kpt = 1;
   int qpt = 2;

   // -- unpack shape --
   int nbatch = dists.size(0);
   int nheads = dists.size(1);
   int nqueries = dists.size(2);
   int k0 = dists.size(3);

   // -- num threads --
   int nthreads_k0 = (k0-1)/kpt+1;
   dim3 nthreads(nthreads_k0);

   int rem_blocks = (65535-1)/nheads+1;
   int nquery_blocks = ((nqueries - 1) / qpt) + 1;
   nquery_blocks = min(nquery_blocks,rem_blocks);
   qpt = ((nqueries - 1) / nquery_blocks) + 1;
   dim3 nblocks(nbatch,nheads,nquery_blocks);

   // fprintf(stdout,"nthreads_k0: %d\n",nthreads_k0);
   // fprintf(stdout,"nbatch,nheads,nquery_blocks: %d,%d,%d\n",
   //         nbatch,nheads,nquery_blocks);
   // fprintf(stdout,"qpt,nquery_blocks,w_threads: %d,%d,%d,%d\n",
   //         qpt,nquery_blocks,ws_h_threads,ws_w_threads);
   // fprintf(stdout,"reflect_bounds,search_abs,anchor_self: %d,%d,%d\n",
   //         reflect_bounds,search_abs,anchor_self);
    
   // launch kernel
   AT_DISPATCH_FLOATING_TYPES(vid0.type(),
                              "prod_dists_forward_kernel", ([&] {
      prod_dists_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        dists.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
        self_dists.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        qstart, stride0, n_h0, n_w0, h0_off, w0_off, h1_off, w1_off,
        ps, pt, chnls, dilation, stride1, use_adj,
        reflect_bounds, search_abs, anchor_self, qpt, kpt);
      }));
}


/****************************

       Backward Pass

****************************/

// same as search