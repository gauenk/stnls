
# -- python --
import cv2,tqdm,copy,pytest
import numpy as np
import unittest
import tempfile
import sys
import shutil
from pathlib import Path
from easydict import EasyDict as edict

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- dnls --
import dnls
import dnls.utils.gpu_mem as gpu_mem
from dnls.utils.pads import comp_pads

# -- check if reordered --
from scipy import optimize
SAVE_DIR = Path("./output/tests/")

#
# -- meshgrid --
#

def pytest_generate_tests(metafunc):
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)
    # test_lists = {"ps":[3],"stride":[1],"dilation":[1,2],
    #               "top":[3],"btm":[62],"left":[2],"right":[62]}
    # test_lists = {"ps":[4],"stride":[1,2],"dilation":[2],
    #               "top":[4],"btm":[64],"left":[1],"right":[61]}
    # test_lists = {"ps":[3,4,5,6,7,8],"stride":[1,2,3,4,5],"dilation":[1,2,3,4,5],
    #               "top":[3],"btm":[57],"left":[7],"right":[57]}
    # test_lists = {"ps":[3],"stride":[2],"dilation":[2],
    #               "top":[3],"btm":[57],"left":[7],"right":[57]}
    test_lists = {"ps":[7],"stride":[4],"dilation":[1],"wt":[0],
                  "ws":[-1],"top":[0],"btm":[64],"left":[0],"right":[64],"k":[-1,5],
                  "exact":[True]}
    # test_lists = {"ps":[3,4,5,6,7,8],"stride":[1,2,3,4,5,8],"dilation":[1,2,3,4,5,8],
    #               "top":[1,11],"btm":[50,57],"left":[3,7],"right":[57,30]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)


#
# -- forward testing --
#

def test_cu_vs_th_fwd(ps,stride,dilation,exact):
    """

    Test the CUDA code with torch code

    Forward Pass

    """
    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    k,pt = 1,1
    wt = 0
    ws = -1
    k = -1
    stride0 = stride
    stride1 = 1
    search_abs = True
    use_k = k>0
    exact = True

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    adj = True
    reflect_bounds = False
    only_full = True

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device)[:1,].contiguous()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-2)
    vid = th.cat([vid,vid],-2)

    # -- normalize --
    vid /= vid.max()
    # vidr = th.ones_like(vid)
    vidr = th.rand_like(vid)

    # -- compute flow --
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape
    chnls = vid.shape[1]

    # -- sub square --
    top,btm,left,right = 0,h,0,w
    coords = [top,left,btm,right]
    # sq_h = coords[2] - coords[0]
    # sq_w = coords[3] - coords[1]

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid.shape, ps, stride0, dil)
    oh1,ow1,_,_ = comp_pads(vid.shape, ps, stride1, dil)
    n_h = (hp - (ps-1)*dil - 1)//stride0 + 1
    n_w = (wp - (ps-1)*dil - 1)//stride0 + 1

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h * n_w
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- exec fold fxns --
    use_adj = True
    h0_off,w0_off,_,_ = comp_pads(vid.shape, ps, stride0, 1)
    h1_off,w1_off,_,_ = comp_pads(vid.shape, ps, stride1, 1)
    search = dnls.search.SearchNl(flows.fflow, flows.bflow, k, ps, pt,
                                  ws, wt, dilation=1, stride=stride1,
                                  use_k = use_k,use_adj=use_adj,
                                  reflect_bounds=reflect_bounds,
                                  search_abs=search_abs,exact=exact,
                                  h0_off=h0_off,w0_off=w0_off,
                                  h1_off=h1_off,w1_off=w1_off)

    # -- query inds --
    qindex = 0
    iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch,stride0,
                                                coords,t,device)

    # -- run search --
    mode = "reflect" if reflect_bounds else "zero"
    score_gt = dnls.simple.search_nn.run_nn(vid,ps,stride=stride0,mode=mode,
                                              dilation=dil,vid1=vidr)
    score_gt = rearrange(score_gt,'(sh sw) (h w) -> h w sh sw',sh=n_h,h=h)


    # -- testing code --
    score_te,inds_te = search(vid,iqueries,vid1=vidr)
    score_te = rearrange(score_te,'(sh sw) (h w) -> h w sh sw',sh=n_h,h=h)

    # -- compare --
    tol = 1e-5
    error = th.mean(th.abs(score_te - score_gt)/score_gt.abs()).item()
    if error > tol: print("error: ",error)
    assert error < tol

    tol = 1e-4
    max_error = th.abs((score_te - score_gt)/score_gt.abs()).max().item()
    if max_error > tol: print("max error: ",max_error)
    assert max_error < tol

def test_cu_vs_simp_fwd(ps,stride,dilation,exact):


    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    k,pt = 1,1
    wt = 0
    ws = -1
    k = -1
    stride0 = stride
    stride1 = 1
    search_abs = True
    use_k = k>0
    exact = True
    use_adj = True

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    adj = True
    reflect_bounds = False
    only_full = True

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device)[:1,].contiguous()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-2)
    vid = th.cat([vid,vid],-2)

    # -- normalize --
    vid /= vid.max()
    # vidr = th.ones_like(vid)
    vidr = th.rand_like(vid)

    # -- compute flow --
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape
    chnls = vid.shape[1]

    # -- sub square --
    top,btm,left,right = 0,h,0,w
    coords = [top,left,btm,right]
    # sq_h = coords[2] - coords[0]
    # sq_w = coords[3] - coords[1]

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid.shape, ps, stride0, dil)
    oh1,ow1,_,_ = comp_pads(vid.shape, ps, stride1, dil)
    n_h = (hp - (ps-1)*dil - 1)//stride0 + 1
    n_w = (wp - (ps-1)*dil - 1)//stride0 + 1

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h * n_w
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- exec fold fxns --
    h0_off,w0_off,_,_ = comp_pads(vid.shape, ps, stride0, 1)
    h1_off,w1_off,_,_ = comp_pads(vid.shape, ps, stride1, 1)
    search = dnls.search.SearchNl(flows.fflow, flows.bflow, k, ps, pt,
                                  ws, wt, dilation=1, stride=stride1,
                                  use_k = use_k,use_adj=use_adj,
                                  reflect_bounds=reflect_bounds,
                                  search_abs=search_abs,exact=exact,
                                  h0_off=h0_off,w0_off=w0_off,
                                  h1_off=h1_off,w1_off=w1_off)

    # -- query inds --
    qindex = 0
    iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch,stride0,
                                                coords,t,device)

    # -- run search --
    score_gt,_ = dnls.simple.search.run(vid,iqueries,flows,k,ps,pt,ws,wt,chnls,
                                        use_adj=use_adj,
                                        reflect_bounds=reflect_bounds,
                                        search_abs=search_abs,
                                        h0_off=h0_off,w0_off=w0_off,
                                        h1_off=h1_off,w1_off=w1_off,
                                        vid1=vidr)
    score_gt = rearrange(score_gt,'(sh sw) (h w) -> h w sh sw',sh=n_h,h=h)


    # -- testing code --
    score_te,inds_te = search(vid,iqueries,vid1=vidr)
    score_te = rearrange(score_te,'(sh sw) (h w) -> h w sh sw',sh=n_h,h=h)

    # -- compare --
    tol = 1e-5
    error = th.mean(th.abs(score_te - score_gt)/score_gt.abs()).item()
    if error > tol: print("error: ",error)
    assert error < tol

    tol = 1e-4
    max_error = th.abs((score_te - score_gt)/score_gt.abs()).max().item()
    if max_error > tol: print("max error: ",max_error)
    assert max_error < tol

#
# -- Backward Testing --
#


def test_cu_vs_th_bwd(ps,stride,dilation,exact):
    """

    Test the CUDA code with torch code

    Backward Pass

    """
    # -- get args --
    dil = dilation
    dname,ext = "davis_baseball_64x64","jpg"
    k,pt = 1,1
    wt = 0
    ws = -1
    k = -1
    stride0 = stride
    stride1 = 1
    search_abs = True
    use_k = k>0
    exact = True

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    adj = True
    reflect_bounds = False
    only_full = True

    # -- load data --
    vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
    vid = th.from_numpy(vid).to(device)[:1,].contiguous()
    gpu_mem.print_gpu_stats(gpu_stats,"post-io")

    # -- grow img --
    vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-1)
    # vid = th.cat([vid,vid],-2)
    vid = th.cat([vid,vid],-2)

    # -- normalize --
    vid /= vid.max()
    # vidr = th.ones_like(vid)
    vidr = th.rand_like(vid)

    # -- allow for grads --
    vid0_te = vid.clone()
    vid1_te = vidr.clone()
    vid0_gt = vid.clone()
    vid1_gt = vidr.clone()
    vid0_te.requires_grad_(True)
    vid1_te.requires_grad_(True)
    vid0_gt.requires_grad_(True)
    vid1_gt.requires_grad_(True)

    # -- compute flow --
    flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

    # -- unpack image --
    device = vid.device
    shape = vid.shape
    t,color,h,w = shape
    vshape = vid.shape
    chnls = vid.shape[1]

    # -- sub square --
    top,btm,left,right = 0,h,0,w
    coords = [top,left,btm,right]
    # sq_h = coords[2] - coords[0]
    # sq_w = coords[3] - coords[1]

    # -- pads --
    oh0,ow0,hp,wp = comp_pads(vid.shape, ps, stride0, dil)
    oh1,ow1,_,_ = comp_pads(vid.shape, ps, stride1, dil)
    n_h = (hp - (ps-1)*dil - 1)//stride0 + 1
    n_w = (wp - (ps-1)*dil - 1)//stride0 + 1

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h * n_w
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- exec fold fxns --
    use_adj = True
    h0_off,w0_off,_,_ = comp_pads(vid.shape, ps, stride0, 1)
    h1_off,w1_off,_,_ = comp_pads(vid.shape, ps, stride1, 1)
    search = dnls.search.SearchNl(flows.fflow, flows.bflow, k, ps, pt,
                                  ws, wt, dilation=1, stride=stride1,
                                  use_k = use_k,use_adj=use_adj,
                                  reflect_bounds=reflect_bounds,
                                  search_abs=search_abs,exact=exact,
                                  h0_off=h0_off,w0_off=w0_off,
                                  h1_off=h1_off,w1_off=w1_off)

    # -- query inds --
    qindex = 0
    iqueries = dnls.utils.inds.get_iquery_batch(qindex,nbatch,stride0,
                                                coords,t,device)
    # -- run search --
    score_te,inds_te = search(vid0_te,iqueries,vid1=vid1_te)
    score_te = rearrange(score_te,'(sh sw) (h w) -> h w sh sw',sh=n_h,h=h)

    # -- comparison --
    mode = "reflect" if reflect_bounds else "zero"
    score_gt = dnls.simple.search_nn.run_nn(vid0_gt,ps,stride=stride0,mode=mode,
                                            dilation=dil,vid1=vid1_gt)
    score_gt = rearrange(score_gt,'(sh sw) (h w) -> h w sh sw',sh=n_h,h=h)

    # -- compute gradient --
    score_grad = th.rand_like(score_gt)
    th.autograd.backward(score_gt,score_grad)
    th.autograd.backward(score_te,score_grad)

    # -- unpack grads --
    grad0_te = vid0_te.grad
    grad1_te = vid1_te.grad
    grad0_gt = vid0_gt.grad
    grad1_gt = vid1_gt.grad


    #
    # -- Backward Step --
    #

    # -- tolerances --
    tol_mean = 1e-5
    tol_max = 1e-4

    # -- check 0 --
    diff = th.abs((grad0_te - grad0_gt)/(grad0_gt.abs()+1e-5))
    error = diff.mean().item()
    assert error < tol_mean
    error = diff.max().item()
    assert error < tol_max

    # -- check 1 --
    diff = th.abs((grad1_te - grad1_gt)/(grad1_gt.abs()+1e-5))
    error = diff.mean().item()
    assert error < tol_mean
    error = diff.max().item()
    assert error < tol_max


# class TestTopKSearch(unittest.TestCase):

#     #
#     # -- Load Data --
#     #

#     def do_load_data(self,dname,sigma,device="cuda:0"):

#         #  -- Read Data (Image & VNLB-C++ Results) --
#         ext = "jpg"
#         vid = dnls.testing.data.load_burst("./data/",dname,ext=ext)
#         clean = th.from_numpy(vid).to(device).contiguous()
#         clean = clean * 1.0
#         noisy = clean + sigma * th.normal(0,1,size=clean.shape,device=device)
#         return clean,noisy

#     def do_load_flow(self,comp_flow,burst,sigma,device):
#         #  -- Empty shells --
#         t,c,h,w = burst.shape
#         tf32,tfl = th.float32,th.long
#         fflow = th.zeros(t,2,h,w,dtype=tf32,device=device)
#         bflow = fflow.clone()

#         # -- pack --
#         flows = edict()
#         flows.fflow = fflow
#         flows.bflow = bflow
#         return flows


#     def init_topk_shells(self,bsize,k,device):
#         tf32,ti32 = th.float32,th.int32
#         vals = float("inf") * th.ones((bsize,k),dtype=tf32,device=device)
#         inds = -th.ones((bsize,k),dtype=ti32,device=device)
#         return vals,inds

#     #
#     # -- [Exec] Sim Search --
#     #

#     def run_comparison(self,dname,sigma,args):

#         # -- get data --
#         noisy,clean = self.do_load_data(dname,sigma)

#         # -- fixed testing params --
#         k = 15
#         BSIZE = 50
#         NBATCHES = 3
#         shape = noisy.shape
#         device = noisy.device
#         t,c,h,w = noisy.shape
#         npix = h*w

#         # -- create empty bufs --
#         bufs = edict()
#         bufs.patches = None
#         bufs.dists = None
#         bufs.inds = None

#         # -- batching info --
#         device = noisy.device
#         shape = noisy.shape
#         t,c,h,w = shape
#         npix_t = h * w
#         qStride = 1
#         qSearchTotal_t = npix_t // qStride # _not_ a DivUp
#         qSearchTotal = t * qSearchTotal_t
#         qSearch = qSearchTotal
#         nbatches = (qSearchTotal - 1) // qSearch + 1

#         # -- unpack --
#         ps = args.ps
#         pt = args.pt
#         ws = args.ws
#         wt = args.wt
#         chnls = args.chnls

#         # -- flows --
#         comp_flow = True
#         clean_flow = True
#         flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,
#                                            noisy,clean,sigma)

#         # -- final args --
#         args.c = c
#         args['stype'] = "faiss"
#         args['queryStride'] = 7
#         args['bstride'] = args['queryStride']

#         # -- exec over batches --
#         for index in range(NBATCHES):

#             # -- new image --
#             clean = th.rand_like(clean).type(th.float32)

#             # -- queries --
#             index = 0
#             queryInds = dnls.utils.inds.get_query_batch(index,qSearch,qStride,
#                                                         t,h,w,device)

#             # -- search using python code --
#             nlDists_simp,nlInds_simp = dnls.simple.search.run(clean,queryInds,
#                                                               flows,k,ps,pt,ws,wt,chnls)

#             # -- search using CUDA code --
#             dnls_search = dnls.search.SearchNl(flows.fflow, flows.bflow, k, ps, pt,
#                                                ws, wt, chnls=chnls,dilation=1, stride=1)
#             nlDists_cu,nlInds_cu = dnls_search(clean,queryInds)

#             # -- to numpy --
#             nlDists_cu = nlDists_cu.cpu().numpy()
#             nlDists_simp = nlDists_simp.cpu().numpy()
#             nlInds_cu = nlInds_cu.cpu().numpy()
#             nlInds_simp = nlInds_simp.cpu().numpy()

#             # -- save mask --
#             dists_cu = rearrange(nlDists_cu,'(t h w) k -> t k h w ',t=t,h=h,w=w)
#             dists_simp = rearrange(nlDists_simp,'(t h w) k -> t k h w ',t=t,h=h,w=w)
#             dists = np.abs(dists_cu - dists_simp)
#             for ti in range(t):
#                 dists_ti = repeat(dists[ti,:,:,:],'t h w -> t c h w ',c=3)
#                 if dists_ti.max() > 1e-3: dists_ti /= dists_ti.max()
#                 dnls.testing.data.save_burst(dists_ti,SAVE_DIR,"dists_%d" % ti)

#             # -- allow for swapping of "close" values --
#             np.testing.assert_array_almost_equal(nlDists_cu,nlDists_simp,5)

#             # -- mostly the same inds --
#             perc_neq = (np.abs(nlInds_cu != nlInds_simp)*1.).mean().item()
#             assert perc_neq < 0.05

#     def test_sim_search(self):

#         # -- init save path --
#         np.random.seed(123)
#         save_dir = SAVE_DIR
#         if not save_dir.exists():
#             save_dir.mkdir(parents=True)

#         # -- test 1 --
#         sigma = 25.
#         dname = "davis_baseball_64x64"
#         # dname = "text_tourbus_64"
#         args = edict({'ps':7,'pt':1,"ws":10,"wt":10,"chnls":1})
#         nreps = 3
#         for r in range(nreps):
#             self.run_comparison(dname,sigma,args)

#     # @pytest.mark.skip()
#     def test_sim_search_fwd_bwd(self):

#         # -- init save path --
#         np.random.seed(123)
#         save_dir = SAVE_DIR
#         if not save_dir.exists():
#             save_dir.mkdir(parents=True)

#         # -- test 1 --
#         sigma = 25.
#         # dname = "text_tourbus_64"
#         dname = "davis_baseball_64x64"

#         # -- get data --
#         noisy,clean = self.do_load_data(dname,sigma)

#         # -- fixed testing params --
#         k = 15
#         BSIZE = 50
#         NBATCHES = 3
#         shape = noisy.shape
#         device = noisy.device
#         t,c,h,w = noisy.shape
#         npix = h*w

#         # -- create empty bufs --
#         bufs = edict()
#         bufs.patches = None
#         bufs.dists = None
#         bufs.inds = None

#         # -- batching info --
#         device = noisy.device
#         shape = noisy.shape
#         t,c,h,w = shape
#         npix_t = h * w
#         qStride = 1
#         qSearchTotal_t = npix_t // qStride # _not_ a DivUp
#         qSearchTotal = t * qSearchTotal_t
#         qSearch = qSearchTotal
#         nbatches = (qSearchTotal - 1) // qSearch + 1

#         # -- unpack --
#         ps = 5
#         pt = 1
#         ws = 5
#         wt = 2
#         chnls = 1

#         # -- flows --
#         comp_flow = True
#         clean_flow = True
#         flows = dnls.testing.flow.get_flow(comp_flow,clean_flow,
#                                            noisy,clean,sigma)

#         # -- new image --
#         clean = th.rand_like(clean).type(th.float32)
#         clean.requires_grad_(True)

#         # -- queries --
#         index = 0
#         queryInds = dnls.utils.inds.get_query_batch(index,qSearch,qStride,
#                                                     t,h,w,device)

#         # -- search using CUDA code --
#         dnls_search = dnls.search.SearchNl(flows.fflow, flows.bflow, k, ps, pt,
#                                            ws, wt, dilation=1, stride=1)
#         nlDists,nlInds = dnls_search(clean,queryInds)
#         ones = th.rand_like(nlDists)
#         loss = th.sum((nlDists - ones)**2)
#         loss.backward()


