"""

Show the error maps for non-local patch aggregation (FoldK)

"""


def run_exp(cfg):

    nreps = cfg.nreps
    use_rand = cfg.use_rand
    exact = cfg.exact
    fold = dnls.FoldK(clean.shape,use_rand=use_rand,nreps=nreps,
                      exact=exact,device=device)


def main():

    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    cache_name = "test_rgb_net" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # cache.clear()

    # -- get defaults --
    cfg = configs.default_test_vid_cfg()
    cfg.isize = "256_256"
    # cfg.isize = "none"
    cfg.bw = True
    cfg.nframes = 10
    cfg.frame_start = 10
    cfg.frame_end = cfg.frame_start+cfg.nframes-1

    # -- get mesh --
    internal_adapt_nsteps = [300]
    internal_adapt_nepochs = [0]
    ws,wt,k,bs,stride = [15],[3],[7],[256],[5]
    dnames,sigmas,use_train = ["set8"],[50.],["false"]
    vid_names = ["snowboard","sunflower","tractor","motorbike",
                 "hypersmooth","park_joy","rafting","touchdown"]
    flow,isizes,adapt_mtypes = ["true"],["none"],["rand"]
    model_names = ["refactored"]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "internal_adapt_nsteps":internal_adapt_nsteps,
                 "internal_adapt_nepochs":internal_adapt_nepochs,
                 "flow":flow,"ws":ws,"wt":wt,"adapt_mtype":adapt_mtypes,
                 "isize":isizes,"use_train":use_train,"stride":stride,
                 "ws":ws,"wt":wt,"k":k, "bs":bs, "model_name":model_names}
    exps = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_a,cfg) # merge the two

    # -- run exps --
    nexps = len(exps)
    for exp_num,exp in enumerate(exps):

        # -- info --
        if verbose:
            print("-="*25+"-")
            print(f"Running experiment number {exp_num+1}/{nexps}")
            print("-="*25+"-")
            pp.pprint(exp)

        # -- logic --
        uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid
        if exp.model_name == "refactored":
            cache.clear_exp(uuid)
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)


if __name__ == "__main__":
    main()
