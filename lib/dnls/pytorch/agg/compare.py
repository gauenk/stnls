

# -- linalg --
import torch as th

# -- flow --
import dnls
from dev_basics import flow

# -- timing --
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt
from dev_basics.utils.misc import set_seed

# -- local --
from . import api


def run(cfg):

    # -- init data --
    set_seed(cfg.seed)
    device = "cuda:0"
    F = cfg.nftrs_per_head * cfg.nheads
    vid = th.randn((1,cfg.nframes,F,cfg.H,cfg.W),device=device,dtype=th.float32)
    flows = flow.orun(vid,False)
    aflows = dnls.nn.ofa.run(flows,stride0=cfg.stride0)

    # -- get the inds --
    nl_search = api.nl.init(cfg)
    _,inds = nl_search(vid)
    th.cuda.synchronize()

    # -- init --
    memer = GpuMemer(True)
    timer = ExpTimer(True)

    # -- get info --
    res = {"flops":[],"radius":[],"time":[],"mem_res":[],"mem_alloc":[],"radius":[]}
    kwargs = {"inds":inds,"flows":flows,"aflows":aflows}
    name = cfg.search_name
    search_fxn = api.init(cfg)
    # search_fxn.set_flows(vid,flows,aflows)
    res['flops'] = search_fxn.flops(1,F,cfg.H,cfg.W)/(1.*10**9)
    res['radius'] = search_fxn.radius(cfg.H,cfg.W)
    th.cuda.synchronize()
    search_fxn(vid,**kwargs) # burn-in
    th.cuda.synchronize()
    with MemIt(memer,name):
        with TimeIt(timer,name):
            search_fxn(vid,**kwargs)
    res['time'] = timer[name]
    res['mem_res'] = memer[name]['res']
    res['mem_alloc'] = memer[name]['alloc']

    # -- copy to res --
    for key in res:
        if key == "name": continue
        res[key] = res[key]
    print(res)

    return res
