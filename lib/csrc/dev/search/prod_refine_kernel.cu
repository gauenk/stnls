
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

/****************************

       Forward Pass

****************************/

template <typename scalar_t>
__global__ void prod_refine_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid0,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid1,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> dists,
    torch::PackedTensorAccessor32<int,7,torch::RestrictPtrTraits> inds,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> self_dists,
    const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> qinds,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int ws_h, int ws_w,
    int ws_h_b, int ws_w_b, int chnls, int dilation,
    int stride1, bool use_adj, bool reflect_bounds, bool search_abs,
    bool full_ws, bool anchor_self, bool use_self,
    int ksearch_iters, int wh_iters, int ww_iters, int qpt){

  // shapes
  int nheads,nframes,color,height,width;
  // bsize = vid0.size(0);
  nheads = vid0.size(1);
  nframes = vid0.size(2);
  // height = vid0.size(3);
  // width = vid0.size(4);
  // color = vid0.size(5);
  color = vid0.size(3);
  height = vid0.size(4);
  width = vid0.size(5);
  int n_hw0 = n_h0 * n_w0;
  int nqueries = dists.size(2);
  int ksearch = qinds.size(3);

  // constants
  float nan = __int_as_float(0xffe00000);
  float inf = __int_as_float(0x7f800000);

  // offsets
  int psHalf = (ps)/2;
  int wsHalf_h = (ws_h)/2;
  int wsHalf_w = (ws_w)/2;
  int wsMax_h = stride1*(ws_h-1-wsHalf_h);
  int wsMax_w = stride1*(ws_w-1-wsHalf_w);
  int wsOff_h = 0;//ws_h/2;
  int wsOff_w = 0;//ws_w/2;  
  int adj = use_adj ? psHalf : 0;

  // cuda index
  int ibatch = blockIdx.x;
  int ihead = blockIdx.y;
  int qi,si,wh,ww;
  int qindex,i_mod;

  // qinds 
  int hj_b,wj_b;

  // decls
  int ti,hi,wi;
  int n_ti,n_hi,n_wi;
  int vH,vW,vT,nH,nW,nT;
  bool valid,vvalid,nvalid;
  bool valid_anchor,valid_n;
  bool vvalid_t,vvalid_h,vvalid_w;
  bool nvalid_t,nvalid_h,nvalid_w;
  bool eq_dim;//eq_ti,eq_hi,eq_wi,
  scalar_t dist,v_pix,n_pix;

  for (int _qi = 0; _qi < qpt; _qi++){
    
    //---------------------------
    //   extract anchor pixel
    //---------------------------

    // -- block start --
    qi = _qi + blockIdx.z*qpt;
    if (qi >= nqueries){ continue; }

    // -- unpack pixel locs --
    qindex = qi + qstart;
    i_mod = qindex % n_hw0;
    ti = qindex / n_hw0;
    wi = ((i_mod % n_w0) * stride0) % width ;
    hi = ((i_mod / n_w0) * stride0) % height;

    // -- valid (anchor pixel) --
    valid_anchor = (ti < nframes) && (ti >= 0);
    valid_anchor = valid_anchor && (hi < height) && (hi >= 0);
    valid_anchor = valid_anchor && (wi < width) && (wi >= 0);
    // valid_anchor = valid_ti && valid_hi && valid_wi;

    // -- search offset --
    if(full_ws){
      wsOff_h = (hi-max(hi-stride1*wsHalf_h,0))/stride1;
      wsOff_w = (wi-max(wi-stride1*wsHalf_w,0))/stride1;
      if ((hi+wsMax_h) >= height){
        wsOff_h+=(hi+wsMax_h-min(hi+stride1*wsMax_h,height-1)-1)/stride1 + 1;
      }
      if ((wi+wsMax_w) >= width){
        wsOff_w+=(wi+wsMax_w-min(wi+stride1*wsMax_w,width-1)-1)/stride1 + 1;
      }
    }else{
      wsOff_h = wsHalf_h;
      wsOff_w = wsHalf_w;
    }

    // ---------------------------------------
    //     for each neighbor in k_search
    // ---------------------------------------
    for(int _si = 0; _si < ksearch_iters; _si++){
      si = threadIdx.x + blockDim.x*_si;
      if (si >= ksearch){ continue; }

      // -- unpack base -- 
      int n_ti = qinds[ibatch][ihead][qi][si][0]; // no search
      int hj_b = qinds[ibatch][ihead][qi][si][1];
      int wj_b = qinds[ibatch][ihead][qi][si][2];

      // ---------------------------------------
      //     for each position to search
      // ---------------------------------------
      for(int _wh = 0; _wh < wh_iters; _wh++){
        wh = threadIdx.y + blockDim.y*_wh;
        if (wh >= ws_h){ continue; }

        for(int _ww = 0; _ww < ww_iters; _ww++){
          ww = threadIdx.z + blockDim.z*_ww;
          if (ww >= ws_w){ continue; }

          // --------------------
          //      init dists
          // --------------------
          dist = 0;

          // ----------------------
          //    spatial center
          // ----------------------
          n_hi = (hj_b) + stride1 * (wh - wsOff_h);
          n_wi = (wj_b) + stride1 * (ww - wsOff_w);

          // ---------------------------
          //      valid (search "n")
          // ---------------------------
          valid_n = (abs(n_hi - hi) <= ws_h_b);
          // if ((hi == 0) && (wi==0)){
          //   printf("n_hi,hi,ws_h_og,valid_n: %d,%d,%d,%d\n",n_hi,hi,ws_h_og,valid_n);
          // }
          valid_n = (abs(n_wi - wi) <= ws_w_b) && valid_n;
          // valid_n = true;
          valid_n = (n_ti < nframes) && (n_ti >= 0) && valid_n;
          valid_n = valid_n && (n_hi < height) && (n_hi >= 0);
          valid_n = valid_n && (n_wi < width) && (n_wi >= 0);
          valid = valid_n && valid_anchor;

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
                  v_pix = vvalid ? vid0[ibatch][ihead][vT][ci][vH][vW] : (scalar_t)0.;
                  n_pix = nvalid ? vid1[ibatch][ihead][nT][ci][nH][nW] : (scalar_t)0.;

                  // -- compute dist --
                  dist += v_pix * n_pix;

                } // ci
              } // pj
            } // pi
          } // pk

          // -- dists --
          if (!valid){ dist = nan; }
          dists[ibatch][ihead][qi][si][wh][ww] = dist;

          // -- inds --
          inds[ibatch][ihead][qi][si][wh][ww][0] = n_ti;
          inds[ibatch][ihead][qi][si][wh][ww][1] = n_hi;
          inds[ibatch][ihead][qi][si][wh][ww][2] = n_wi;

          // -- final check [put self@index 0] --
          if (anchor_self){
            eq_dim = n_ti == ti;
            eq_dim = eq_dim && (n_hi == hi);
            eq_dim = eq_dim && (n_wi == wi);
            // eq_dim = eq_ti && eq_hi && eq_wi;
            if (eq_dim && use_self){
              self_dists[ibatch][ihead][qi] = dist; // update self
              dists[ibatch][ihead][qi][si][wh][ww] = inf;
            }else if (eq_dim){
              dists[ibatch][ihead][qi][si][wh][ww] = inf;
            }
          }

        } //  ww
      } // wh
    } // si
  } // qi
} // fxn

void prod_refine_forward_cuda(
    const torch::Tensor vid0, const torch::Tensor vid1,
    torch::Tensor dists, torch::Tensor inds,
    torch::Tensor self_dists, torch::Tensor qinds,
    int qstart, int stride0, int n_h0, int n_w0,
    int h0_off, int w0_off, int h1_off, int w1_off,
    int ps, int pt, int ws_h, int ws_w,
    int ws_h_og,int ws_w_og,
    int chnls, int dilation, int stride1,
    bool use_adj, bool reflect_bounds, bool search_abs,
    bool full_ws, bool anchor_self, bool use_self){

   // fprintf(stdout,"ws_h_og,ws_w_og: %d,%d\n",ws_h_og,ws_w_og);
   // fprintf(stdout,"ws_h,ws_w: %d,%d\n",ws_h,ws_w);
   // fprintf(stdout,"n_h0,n_w0: %d,%d\n",n_h0,n_w0);
   // fprintf(stdout,"qstart, nqueries: %d,%d\n",qstart,nqueries);
   // launch params
   // our many (too many?) registers limit the number of threads
   int nheads = dists.size(1);
   int nqueries = dists.size(2);
   int ksearch = inds.size(3);
   int ksearch_threads = std::min(ksearch,12);
   int ws_h_threads = std::min(ws_h,8);
   int ws_w_threads = std::min(ws_w,8);
   int ksearch_iters = ((ksearch-1)/ksearch_threads)+1;
   int ws_h_iters = ((ws_h-1)/ws_h_threads) + 1;
   int ws_w_iters = ((ws_w-1)/ws_w_threads) + 1;
   dim3 nthreads(ksearch_threads,ws_h_threads,ws_w_threads);

   int bsize = vid0.size(0);
   int rem_blocks = (65535-1)/nheads+1;
   int qpt = 2;
   int nquery_blocks = ((nqueries - 1) / qpt) + 1;
   nquery_blocks = min(nquery_blocks,rem_blocks);
   qpt = ((nqueries - 1) / nquery_blocks) + 1;
   dim3 nblocks(bsize,nheads,nquery_blocks);


   // -- search bounds --
   int ws_h_b = ws_h_og/2;
   int ws_w_b = ws_w_og/2;

   // -- info --
   // fprintf(stdout,"bsize,nheads,nquery_blocks: %d,%d,%d\n",
   //         bsize,nheads,nquery_blocks);
   // fprintf(stdout,"qpt,nquery_blocks,w_threads: %d,%d,%d,%d\n",
   //         qpt,nquery_blocks,ws_h_threads,ws_w_threads);
   // fprintf(stdout,"reflect_bounds,search_abs,full_ws,anchor_self,use_self: %d,%d,%d,%d,%d\n",
   //         reflect_bounds,search_abs,full_ws,anchor_self,use_self);
   // fprintf(stdout,"ws_h_iters,ws_w_iters: %d,%d,%d,%d,\n",ws_h_iters,ws_w_iters,ws_h,ws_w);
    
   // launch kernel
   AT_DISPATCH_FLOATING_TYPES(vid0.type(),"prod_refine_forward_kernel", ([&] {
      prod_refine_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        vid0.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        vid1.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        dists.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        inds.packed_accessor32<int,7,torch::RestrictPtrTraits>(),
        self_dists.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        qinds.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
        qstart, stride0, n_h0, n_w0, h0_off, w0_off, h1_off, w1_off,
        ps, pt, ws_h, ws_w, ws_h_b, ws_w_b, chnls, dilation, stride1,
        use_adj, reflect_bounds, search_abs, full_ws, anchor_self, use_self,
        ksearch_iters, ws_h_iters, ws_w_iters, qpt);
      }));
}


