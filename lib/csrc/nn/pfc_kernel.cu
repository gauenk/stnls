
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

/****************************

       Helper Funcs

****************************/


#define CUDA_KERNEL_LOOP_TYPE(i, n, index_type)                         \
  int64_t _i_n_d_e_x = blockIdx.x * blockDim.x + threadIdx.x;           \
  for (index_type i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=blockDim.x * gridDim.x, i=_i_n_d_e_x)

#define CUDA_KERNEL_LOOP(i, n) CUDA_KERNEL_LOOP_TYPE(i, n, int)

__inline__ __device__ int bounds(int val, int lb, int ub ){
  int vval = val;
  if (val < lb){
    vval = 2*lb - val;
  }else if (val >= ub){
    vval = 2*(ub-1) - val;
  }
  return vval;
}


/**************************************

       Forward Pass (with Inds)

**************************************/


// template <typename scalar_t>
// __global__ void patch_full_connected_forward_kernel(
//     torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid,
//     torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> vid_in,
//     torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> weights,
//     torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> bias,
//     int qstart, int nqueries, int ps, int top, int left, int btm, int right, 
//     int hw_start, int stride, int dilation, int adj,
//     bool only_full, bool use_reflect, int num_kernels) {


//     // -- unpack --
//     int bsize = vid.size(0);
//     int nframes = vid.size(1);
//     int nftrs_out = vid.size(2);
//     int nftrs_in = vid.size(3);
//     int height = vid.size(4);
//     int width = vid.size(5);
//     int pt = 1;
//     int psOffset = (ps-1)/2;
//     int psHalf = ps/2;
//     int hw = height*width;
//     int fill_pad = psHalf * dilation;
//     int dil = dilation;
//     // int width_s = width/stride;
//     // int hw_s = (height/stride)*(width/stride);
//     // int num_kernels = inds.size(0);
//     bool valid,valid_q;
//     // bool is_edge;
//     // int nhits,nhits_q;
//     // int ndim = ps*ps*pt;

//     // -- weight --
//     int reflect_max = psHalf;
//     scalar_t weight;

//     // -- coords with pads --
//     int pad = dilation*(ps/2);
//     // pad = (adj > 0) ? 0 : pad; // suspect line.
//     int top_p = std::max(top-pad,0);
//     int left_p = std::max(left-pad,0);
//     int btm_p = std::min(btm+pad,height);
//     int right_p = std::min(right+pad,width);

//     // -- get indices --
//     int c_idx1 = blockIdx.y;
//     int ibatch = blockIdx.z;

//     // coords
//     int sq_hp = btm_p - top_p;
//     int sq_wp = right_p - left_p;
//     int sq_hwp = sq_hp * sq_wp;

//     // -- adjust endpoint for "adj" --
//     // no spilling over right-hand boundary
//     int right_a = right - (ps-1)*dil;
//     int btm_a = btm - (ps-1)*dil;
//     int right_bnd = (only_full) ? right_a : right;
//     int btm_bnd = (only_full) ? btm_a : btm;
  
//     // -- make square --
//     int sq_h = btm - top;
//     int sq_w = right - left;
//     int sq_hw = sq_h * sq_w;

//     // -- strided size --
//     int n_h = int((sq_h-1) / stride) + 1;
//     int n_w = int((sq_w-1) / stride) + 1;
//     if (only_full){
//       n_h = (sq_h - (ps-1)*dil - 1)/stride + 1;
//       n_w = (sq_w - (ps-1)*dil - 1)/stride + 1;
//     }
//     int n_hw = n_h * n_w;


//     CUDA_KERNEL_LOOP(_index, num_kernels) {

//       // index to pixel location
//       int index = (_index);
//       const int64_t t_im = (index / sq_hwp);
//       const int64_t i_mod = index % sq_hwp;
//       const int64_t w_im = (i_mod % sq_wp) + left_p;
//       const int64_t h_im = ((i_mod / sq_wp) % sq_hp) + top_p;

//       // Which patches (qi) impact me (t_im,w_im,h_im)?
//       for(int c_idx0 = 0; c_idx0 < nftrs_out; c_idx0++){
//         scalar_t val = 0;
//         int Z = 0;
//         for (int pk = 0; pk < pt; pk++){
//           for (int pi = 0; pi < ps; pi++){
//             for (int pj = 0; pj < ps; pj++){

//               // -- offsets for ni --
//               // use "psOffset" instead of "psHalf" because of reflection.
//               int _hi = h_im + dilation*(pi - psOffset);// - psOffset - adj);
//               int _wi = w_im + dilation*(pj - psOffset);// - psOffset - adj);
//               int ti = t_im + pk;

//               // -- check bounds (we need the patch for the pixel!) --
//               valid = (_hi >= (top+ps/2)) && (_hi < (btm_bnd-ps/2));
//               valid = valid && (_wi >= (left+ps/2)) && (_wi < (right_bnd-ps/2));
//               int hi = use_reflect ? bounds(_hi,top,btm) : _hi;
//               int wi = use_reflect ? bounds(_wi,left,right) : _wi;
//               // valid = (wi >= left) && (wi < right_bnd);
//               // valid = valid && (hi >= top) && (hi < btm_bnd);

//               // -- only if proposed index is aligned with stride --
//               valid = valid && ((hi-top) % stride == 0) && ((wi-left) % stride == 0);

//               // -- compute ni --
//               int qi = ti * n_hw + qstart;
//               qi += (((hi-top)/stride) * n_w);
//               qi += ((wi-left)/stride);
//               qi -= hw_start;

//               // -- patch indexing --
//               int w_ip = ps-1-pi;
//               int h_ip = ps-1-pj;

//               // -- reflect to match --
//               if (_wi > wi){
//                 w_ip = pi;
//                 valid = valid && (w_ip < psHalf);
//               }
//               else if(_wi < wi){
//                 w_ip = pi;
//                 valid = valid && (w_ip > psHalf);
//               }

//               if (_hi > hi){
//                 h_ip = pj;
//                 valid = valid && (h_ip < psHalf);
//               }
//               else if(_hi < hi){
//                 h_ip = pj;
//                 valid = valid && (h_ip > psHalf);
//               }


//               // -- reflect offset --
//               // hi = bounds(_hi+pi,top,btm);
//               // wi = bounds(_wi+pj,left,right);
//               // hi = bounds(h_im+pi-psOffset,top,btm);
//               // wi = bounds(w_im+pj-psOffset,left,right);

//               // -- accumulate --

//               // -- compute the inner product --

//               // valid_q = valid;
//               valid_q = valid && (qi >= 0) && (qi < nqueries);
//               if (valid_q){
//                 Z += 1;

//                 // -- index weight _row_  (h_ip,w_ip) --
//                 int h_c = (hi - psHalf) + h_ip;
//                 int w_c = (wi - psHalf) + w_ip;
//                 //float* weights_r = weights[c_idx1][h_ip][w_ip];
//                 // weights_r[c_idx0][_pi][_pj];

//                 // -- inner product across (hi,wi) patch using weight (h_ip,w_ip) --
//                 scalar_t pval = 0;
//                 for (int _pi = 0; _pi < ps; _pi++){
//                   for (int _pj = 0; _pj < ps; _pj++){

//                     // -- weight index from "_ip" & "_im" index --
//                     // weight = weights[c_idx0][_pi][_pj][c_idx1][h_ip][w_ip];
//                     weight = weights[c_idx0][w_ip][h_ip][c_idx1][_pi][_pj];
//                     // weight = weights[c_idx0][w_ip][h_ip][c_idx1][_pj][_pi];
//                     // weight = weights[c_idx0][_pi][_pj][c_idx1][h_ip][h_ip];

//                     // -- patch --
//                     int hi_in = (hi) + dilation*(_pi - psOffset);// - psOffset - adj);
//                     int wi_in = (wi) + dilation*(_pj - psOffset);// - psOffset - adj);
//                     hi_in = bounds(hi_in,top,btm);
//                     wi_in = bounds(wi_in,left,right);
//                     bool valid_in = (hi_in >= 0) && (hi_in < height);
//                     valid_in = valid_in && (wi_in >= 0) && (wi_in < width);
//                     // bool valid_in = true;

//                     // -- accumulate partial sum --
//                     if (valid_in){
//                       pval += weight * vid_in[ibatch][t_im][c_idx1][hi_in][wi_in];
//                     }
//                     // val += patches[ibatch][qi][0][0][c_out][h_ip][w_ip];
//                   }
//                 }
//                 // pval = (pval > 0) ? pval : 0;
//                 pval += bias[c_idx0][w_ip][h_ip];
//                 val += pval;
//                 // val += bias[c_idx1][h_ip][w_ip];
//               }

//             } // for patch "t"
//           } // for patch size
//         } // for patch size
//         vid[ibatch][t_im][c_idx0][c_idx1][h_im][w_im] = val/Z;
//       } // for nftrs
//     } // for each pixel (with stride)
// }


/****************************

     Allows for ReLu!

****************************/

template <typename scalar_t>
__global__ void patch_full_connected_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> vid,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> vid_in,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> bias,
    int qstart, int nqueries, int ps, int top, int left, int btm, int right, 
    int hw_start, int stride, int dilation, int adj,
    bool only_full, bool use_reflect, int num_kernels) {


    // -- unpack --
    int bsize = vid.size(0);
    int nframes = vid.size(1);
    int nftrs_out = vid.size(2);
    int nftrs_in = vid.size(3);
    int height = vid.size(4);
    int width = vid.size(5);
    int pt = 1;
    int psOffset = (ps-1)/2;
    int psHalf = ps/2;
    int hw = height*width;
    int fill_pad = psHalf * dilation;
    int dil = dilation;
    // int width_s = width/stride;
    // int hw_s = (height/stride)*(width/stride);
    // int num_kernels = inds.size(0);
    bool valid,valid_q;
    // bool is_edge;
    // int nhits,nhits_q;
    // int ndim = ps*ps*pt;

    // -- weight --
    int reflect_max = psHalf;
    scalar_t weight;

    // -- coords with pads --
    int pad = dilation*(ps/2);
    // pad = (adj > 0) ? 0 : pad; // suspect line.
    int top_p = std::max(top-pad,0);
    int left_p = std::max(left-pad,0);
    int btm_p = std::min(btm+pad,height);
    int right_p = std::min(right+pad,width);

    // -- get indices --
    int c_idx0 = blockIdx.y;
    int ibatch = blockIdx.z;

    // coords
    int sq_hp = btm_p - top_p;
    int sq_wp = right_p - left_p;
    int sq_hwp = sq_hp * sq_wp;

    // -- adjust endpoint for "adj" --
    // no spilling over right-hand boundary
    int right_a = right - (ps-1)*dil;
    int btm_a = btm - (ps-1)*dil;
    int right_bnd = (only_full) ? right_a : right;
    int btm_bnd = (only_full) ? btm_a : btm;
  
    // -- make square --
    int sq_h = btm - top;
    int sq_w = right - left;
    int sq_hw = sq_h * sq_w;

    // -- illegal boundary --
    // int ill_bnd = ps/2;
    int ill_bnd = 0;

    // -- strided size --
    int n_h = int((sq_h-1) / stride) + 1;
    int n_w = int((sq_w-1) / stride) + 1;
    if (only_full){
      n_h = (sq_h - (ps-1)*dil - 1)/stride + 1;
      n_w = (sq_w - (ps-1)*dil - 1)/stride + 1;
    }
    int n_hw = n_h * n_w;


    CUDA_KERNEL_LOOP(_index, num_kernels) {

      // index to pixel location
      int index = (_index);
      const int64_t t_im = (index / sq_hwp);
      const int64_t i_mod = index % sq_hwp;
      const int64_t w_im = (i_mod % sq_wp) + left_p;
      const int64_t h_im = ((i_mod / sq_wp) % sq_hp) + top_p;

      // Which patches (qi) impact me (t_im,w_im,h_im)?
      scalar_t val = 0;
      int Z = 0;
      for (int pk = 0; pk < pt; pk++){
        for (int pi = 0; pi < ps; pi++){
          for (int pj = 0; pj < ps; pj++){

            // -- offsets for ni --
            // use "psOffset" instead of "psHalf" because of reflection.
            int _hi = h_im + dilation*(pi - psOffset);// - psOffset - adj);
            int _wi = w_im + dilation*(pj - psOffset);// - psOffset - adj);
            int ti = t_im + pk;

            // -- check bounds (we need the patch for the pixel!) --
            valid = (_hi >= (top+ill_bnd)) && (_hi < (btm_bnd-ill_bnd));
            valid = valid && (_wi >= (left+ill_bnd)) && (_wi < (right_bnd-ill_bnd));
            int hi = use_reflect ? bounds(_hi,top,btm) : _hi;
            int wi = use_reflect ? bounds(_wi,left,right) : _wi;
            // valid = (wi >= left) && (wi < right_bnd);
            // valid = valid && (hi >= top) && (hi < btm_bnd);

            // -- only if proposed index is aligned with stride --
            valid = valid && ((hi-top) % stride == 0) && ((wi-left) % stride == 0);

            // -- compute ni --
            int qi = ti * n_hw + qstart;
            qi += (((hi-top)/stride) * n_w);
            qi += ((wi-left)/stride);
            qi -= hw_start;

            // -- patch indexing --
            int w_ip = ps-1-pi;
            int h_ip = ps-1-pj;

            // -- reflect to match --
            if (_wi > wi){
              w_ip = pi;
              valid = valid && (w_ip < psHalf);
            }
            else if(_wi < wi){
              w_ip = pi;
              valid = valid && (w_ip > psHalf);
            }

            if (_hi > hi){
              h_ip = pj;
              valid = valid && (h_ip < psHalf);
            }
            else if(_hi < hi){
              h_ip = pj;
              valid = valid && (h_ip > psHalf);
            }


            // -- reflect offset --
            // hi = bounds(_hi+pi,top,btm);
            // wi = bounds(_wi+pj,left,right);
            // hi = bounds(h_im+pi-psOffset,top,btm);
            // wi = bounds(w_im+pj-psOffset,left,right);

            // -- accumulate --

            // -- compute the inner product --

            // valid_q = valid;
            valid_q = valid && (qi >= 0) && (qi < nqueries);
            if (valid_q){
              Z += 1;

              // -- index weight _row_  (h_ip,w_ip) --
              int h_c = (hi - psHalf) + h_ip;
              int w_c = (wi - psHalf) + w_ip;
              //float* weights_r = weights[c_idx1][h_ip][w_ip];
              // weights_r[c_idx0][_pi][_pj];

              // -- inner product across (hi,wi) patch using weight (h_ip,w_ip) --
              scalar_t pval = 0;
              for (int _pi = 0; _pi < ps; _pi++){ // ps 
                for (int _pj = 0; _pj < ps; _pj++){ // ps

                  // -- patch --
                  int hi_in = (hi) + dilation*(_pi - psOffset);// - psOffset - adj);
                  int wi_in = (wi) + dilation*(_pj - psOffset);// - psOffset - adj);
                  hi_in = bounds(hi_in,top,btm);
                  wi_in = bounds(wi_in,left,right);
                  bool valid_in = (hi_in >= 0) && (hi_in < height);
                  valid_in = valid_in && (wi_in >= 0) && (wi_in < width);
                  // bool valid_in = true;

                  // -- accumulate partial sum --
                  if (valid_in){
                    for(int c_idx1 = 0; c_idx1 < 1; c_idx1++){ // nftrs_in
                      weight = weights[c_idx0][w_ip][h_ip][c_idx1][_pi][_pj];
                      pval += weight * vid_in[ibatch][t_im][c_idx1][hi_in][wi_in];
                    }
                    // [old ref] val += patches[ibatch][qi][0][0][c_out][h_ip][w_ip];
                  }
                }
              }
              pval += bias[c_idx0][w_ip][h_ip];
              pval = (pval > 0) ? pval : 0;
              val += pval;
              // val += bias[c_idx1][h_ip][w_ip];
            }

          } // for patch "t"
        } // for patch size
      } // for patch size
      vid[ibatch][t_im][c_idx0][0][h_im][w_im] = val/Z;
    } // for each pixel (with stride)
}


void patch_full_connected_forward_cuda(
    torch::Tensor vid, 
    torch::Tensor vid_in,torch::Tensor weights, torch::Tensor bias,
    int qstart, int nqueries, int ps,
    int top, int left, int btm, int right,
    int hw_start, int stride, int dilation, int adj,
    bool only_full, bool use_reflect){

  // batching entire image always
  int bsize = vid.size(0);
  int nframes = vid.size(1);
  int nftrs_out = vid.size(2);
  int nftrs_in = vid.size(3);
  int height = vid.size(4);
  int width = vid.size(5);

  // -- coords with pads --
  int pad = dilation*(ps/2);
  // pad = (adj > 0) ? 0 : pad;
  int top_p = std::max(top-pad,0);
  int left_p = std::max(left-pad,0);
  int btm_p = std::min(btm+pad,height);
  int right_p = std::min(right+pad,width);

  // -- coords --
  int sq_hp = btm_p - top_p;
  int sq_wp = right_p - left_p;
  int sq_hwp = sq_hp * sq_wp;

  // -- launch params --
  int nthreads = 512;
  int num_kernels = nframes*sq_hwp; // ? - qstart?
  int nblocks_queries = (num_kernels-1) / nthreads+1;
  // dim3 nblocks(nblocks_queries,nftrs_in,bsize); // original
  dim3 nblocks(nblocks_queries,nftrs_out,bsize); // origincal
  // only_full = true;

  // fprintf(stdout,"ps,top,left,btm,right,only_full,use_reflect,dilation: \
  //                 %d,%d,%d,%d,%d,%d,%d,%d\n",
  //         ps,top,left,btm,right,only_full,use_reflect,dilation);

  // -- launch kernel --
  AT_DISPATCH_FLOATING_TYPES(vid.type(), "pfc_kernel", ([&] {
    patch_full_connected_forward_kernel<scalar_t><<<nblocks, nthreads>>>(
        vid.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        vid_in.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        bias.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        qstart,nqueries,ps,top,left,btm,right,hw_start,stride,dilation,adj,
        only_full,use_reflect,num_kernels);
      }));
}
