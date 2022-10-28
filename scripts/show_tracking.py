"""

Show how memory size T_s of K_s non-local indices can predict
the future, streamed T_s frames' set of K_s non-local indices. (L > K)

"""

from functools import partial


def run_exp(_cfg):

    # -- init config --
    cfg = copy.deepcopy(_cfg)
    cache_io.exp_strings2bools(cfg)
    configs.set_seed(cfg.seed)
    root = (Path(__file__).parents[0] / ".." ).absolute()

    # -- init timer --
    timer = ExpTimer()

    # -- init model --
    model_cfg = satten.get_model_config(cfg)
    model = satten.load_model(**model_cfg)

    # -- init results --
    # results = edict()
    # results.timer_flow = []
    # results.timer_deno = []

    # -- unpack --
    T_m = cfg.memory_nframes
    K_m = cfg.memory_k
    T_s = cfg.stream_nframes
    K_s = cfg.stream_k

    # -- load data --
    data,loaders = data_hub.load(cfg)
    sample = data.tr[0]
    clean,noisy = sample['clean'],sample['noisy']

    # -- compute non-local search --
    search_cfg = natten.get_search_config(cfg)
    search_cfg.k = K_m
    search_cfg.match_shape = True
    dists_mem,inds_mem = satten.run_nls(noisy[:T_m],**search_cfg)
    print("dists_mem.shape: ",dists_mem.shape) # B x T_m x K_m x 3 x H x W

    # -- estimate next non-local indices & compute quality --
    imodel = satten.get_inds_model(inds_config)
    inds_pred = imodel(inds_mem,dists_mem,T_s,K_s) # satten.predict_inds(
    dists_pred = satten.compute_dists(noisy[T_m:T_m+T_s],pred_inds)

    # -- compute true non-local indices --
    search_cfg.k = K_s
    dists_gt,inds_gt = satten.run_nls(noisy[T_m:T_m+T_s],**search_cfg)

    # -- plot examples --
    locs_pred = {"dists":dists_pred,"inds":inds_pred}
    locs_gt = {"dists":dists_gt,"inds":inds_gt}
    fnames = satten.plot_examples(locs_gt,locs_pred,noisy,T_m,T_s,cfg.save_dir)

    # -- compare predicted vs true --
    dims = list(range(2,dists_gt.ndim))
    dists_error = th.mean((dists_gt - dists_pred)**2,dims=dims).item()
    inds_acc = th.mean(th.all(inds_gt == inds_pred,-1).float(),dims).item()

    # -- denoise [predicted state] --
    with timer("pred"):
        state_pred = edict({"name":"satten","dists":dists_pred,"inds":inds_pred})
        deno_pred = model(noisy[T_m:T_m+T_s],state=state_pred)
        psnr_pred = compute_psnr(deno_pred,clean[T_m:T_m+T_s])

    # -- denoise [ground-truth] --
    with timer("gt"):
        state_gt = edict({"name":"satten","dists":dists_gt,"inds":inds_gt})
        deno_gt = model(noisy[T_m:T_m+T_s],state=state_gt)
        psnr_gt = compute_psnr(deno_pred,clean[T_m:T_m+T_s])

    # -- denoise [orginal model] --
    with timer("og"):
        deno_og = model(noisy[T_m:T_m+T_s],flows=flows,state=None)
        psnr_og = compute_psnr(deno_og,clean[T_m:T_m+T_s])

    # -- info --
    print(fnames)
    print(dists_error,inds_acc)
    print(psnr_pred,psnr_gt,psnr_og)

    # -- aggregate results --
    results = edict()
    results.fnames = fnames
    results.psnr_pred = psnrs_pred
    results.psnr_gt = psnrs_gt
    results.psnr_og = psnrs_og
    results.dists_error = dists_error
    results.inds_acc = inds_acc
    for name,time in timer.items():
        results[name] = time

    return results

def main():
    run_exp(cfg)

