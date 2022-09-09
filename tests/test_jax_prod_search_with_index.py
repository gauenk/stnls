
# -- data mgnmt --
from pathlib import Path

# -- testing --
import pytest

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- dnls --
import dnls
import dnls.utils.gpu_mem as gpu_mem
from dnls.utils.pads import comp_pads
from dnls.utils.inds import get_batching_info

# -- paths --
SAVE_DIR = Path("./output/tests/prod_search")

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


@pytest.mark.jax
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

    # -- init vars --
    device = "cuda:0"
    clean_flow = True
    comp_flow = False
    gpu_stats = False
    adj = False
    reflect_bounds = False

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

    # -- compute flow --
    flows = dnls.flow.get_flow(comp_flow,clean_flow,vid,vid,0.)

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
    _,_,n0,n1 = get_batching_info(vid.shape,stride0,stride1,ps,dil)
    n_h0,n_w0 = n0[0],n0[1]
    n_h1,n_w1 = n1[0],n1[1]

    # -- batching info --
    npix = t * h * w
    ntotal = t * n_h0 * n_w0
    nbatch = ntotal
    nbatches = (ntotal-1) // nbatch + 1

    # -- exec fold fxns --
    use_adj = True
    # oh0, ow0, oh1, ow1 = 0, 0, 0, 0
    # oh0, ow0, oh1, ow1 = -oh0, -ow0, -oh1, -ow1
    search_gt = dnls.search.init("prod_with_index",flows.fflow, flows.bflow,
                              k, ps, pt, ws, wt, oh0, ow0, oh1, ow1,
                              chnls=-1,dilation=dil,
                              stride0=stride0, stride1=stride1,
                              reflect_bounds=reflect_bounds,use_k=False,
                              search_abs=True,use_adj=use_adj,
                              exact=exact)
    search_te = dnls.jax.search.init("prod_with_index",flows.fflow, flows.bflow,
                                     k, ps, pt, ws, wt, oh0, ow0, oh1, ow1,
                                     chnls=-1,dilation=dil,
                                     stride0=stride0, stride1=stride1,
                                     reflect_bounds=reflect_bounds,use_k=False,
                                     search_abs=True,use_adj=use_adj,
                                     exact=exact)
    # -- run search --
    vidr = th.rand_like(vid)
    qindex = 0
    score_gt,inds_gt = search_gt(vid,qindex,nbatch,vid1=vidr)
    score_te,inds_te = search_te(vid,qindex,nbatch,vid1=vidr)

    # -- compare --
    args0 = th.where(th.logical_not(th.isinf(score_gt))) # remove all inf
    diff = th.abs(score_te - score_gt) / (score_gt.abs() + 1e-5)
    diff = diff[args0]

    tol = 1e-5
    error = th.mean(th.abs(score_te - score_gt)).item()
    if error > tol: print("error: ",error)
    assert error < tol

    tol = 1e-4
    max_error = th.abs(score_te - score_gt).max().item()
    if max_error > tol: print("max error: ",max_error)
    assert max_error < tol
