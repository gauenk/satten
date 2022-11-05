"""

Show how memory size T_s of K_s non-local indices can predict
the future, streamed T_s frames' set of K_s non-local indices. (L > K)

"""


# -- python imports --
import os,copy
dcopy = copy.deepcopy
import pprint
import pandas as pd
pp = pprint.PrettyPrinter(indent=4)
from pathlib import Path
from functools import partial
from easydict import EasyDict as edict
from einops import repeat
import torch as th

# -- cache --
import cache_io

# -- data --
import data_hub

# -- optical flow --
from satten import flow

# -- timer --
from satten.utils.timer import ExpTimer,TimeIt

# -- package misc --
import satten
from satten import utils
from satten.utils import optional
from satten.configs import show_tracking as configs
from satten.utils.metrics import compute_psnrs,compute_ssims

def run_exp(_cfg):

    #
    # -- inits --
    #

    # -- config --
    cfg = dcopy(_cfg)
    cache_io.exp_strings2bools(cfg)
    utils.set_seed(cfg.seed)
    root = (Path(__file__).parents[0] / ".." ).absolute()
    device = cfg.device

    # -- timer --
    timer = ExpTimer()

    # -- classifier --
    model_cfg = satten.extract_model_config(cfg.model)
    model = satten.load_model(**model_cfg)

    # -- index prediction --
    imodel_cfg = satten.extract_imodel_config(cfg)
    imodel = satten.load_imodel(**imodel_cfg)

    # -- search --
    search_cfg = satten.extract_search_config(cfg)
    search_cfg.output_as_vid_shape = True
    search = satten.init_search(**search_cfg)

    # -- dists --
    dists_cfg = satten.extract_dists_config(cfg)
    dists_cfg.output_as_vid_shape = True
    dists = satten.init_dists(**dists_cfg)


    # -- init results --
    # results = edict()
    # results.timer_flow = []
    # results.timer_deno = []

    #
    # -- exp start --
    #

    # -- unpack --
    T_m = cfg.memory_nframes
    K_m = cfg.memory_k
    T_s = cfg.stream_nframes
    K_s = cfg.stream_k
    K_search = cfg.k

    # -- load data --
    data,loaders = data_hub.sets.load(cfg)
    frame_start = optional(cfg,"frame_start",0)
    frame_end = optional(cfg,"frame_end",-1)
    indices = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,
                                     frame_start,frame_end)
    sample = data[cfg.dset][indices[0]]
    clean,noisy = sample['clean'][None,:],sample['noisy'][None,:]
    clean,noisy = clean.to(device),noisy.to(device)
    clean,noisy = clean/255.,noisy/255.
    print("clean.shape: ",clean.shape)

    # -- compute non-local search --
    search_cfg.k = K_m
    dists_mem,inds_mem = satten.run_search(noisy[:,:T_m],**search_cfg)
    print("dists_mem.shape: ",dists_mem.shape) # B x T_m x K_m x 3 x H x W

    # -- estimate next non-local indices & compute quality --
    dists_cfg.k = K_s
    inds_pred = imodel(inds_mem,dists_mem)
    print("inds_pred.shape: ",inds_pred.shape)
    dists_pred,inds_pred_topk = satten.run_dists(noisy[:,T_m:T_m+T_s],
                                                 inds_pred,dists_cfg)
    print("dists_pred.shape: ",dists_pred.shape)

    # -- compute true non-local indices --
    search_cfg.k = K_s
    dists_gt,inds_gt = satten.run_search(noisy[:,T_m:T_m+T_s],**search_cfg)
    # print("dists_gt.shape: ",dists_gt.shape)
    # print(inds_gt)
    # exit(0)

    # -- plot examples --
    locs_pred = {"dists":dists_pred,"inds":inds_pred}
    locs_gt = {"dists":dists_gt,"inds":inds_gt}
    fnames = "name"
    # fnames = satten.plot_examples(locs_gt,locs_pred,noisy,T_m,T_s,cfg.save_dir)
    print(dists_gt)
    print(dists_pred)

    # -- compare predicted vs true --
    dims = [0,1,] + list(range(3,dists_gt.ndim))
    dists_error = th.mean((dists_gt - dists_pred)**2,dim=dims)
    print(dists_error.shape)
    print("inds_gt.shape: ",inds_gt.shape)
    print("inds_pred.shape: ",inds_pred.shape)
    inds_acc = th.mean(th.all(inds_gt == inds_pred,-1).float(),dims).item()

    # -- denoise [predicted state] --
    with TimeIt(timer,"pred"):
        state_pred = edict({"name":"satten","dists":dists_pred,"inds":inds_pred})
        deno_pred = model(noisy[T_m:T_m+T_s],state=state_pred)
        psnr_pred = compute_psnr(deno_pred,clean[T_m:T_m+T_s])

    # -- denoise [ground-truth] --
    with TimeIt(timer,"gt"):
        state_gt = edict({"name":"satten","dists":dists_gt,"inds":inds_gt})
        deno_gt = model(noisy[T_m:T_m+T_s],state=state_gt)
        psnr_gt = compute_psnr(deno_pred,clean[T_m:T_m+T_s])

    # -- denoise [orginal model] --
    with TimeIt(timer,"og"):
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

    # -- init info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_name = "show_tracking" # current!
    cache = cache_io.ExpCache(".cache_io",cache_name)
    # cache.clear()

    # -- grab default --
    default_cfg = configs.default()
    default_cfg.cropmode = "center"

    # -- grid --
    isize = ["100_400","none"]
    ps,creps = [7],[1,3,5,10]
    ws,wt,k,nreps = [5],[0],[15],[3]
    exp_lists = {"ps":ps,"ws":ws,"wt":wt,"k":k,"isize":isize,
                 "nreps":nreps,"creps":creps}
    exps = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps,default_cfg) # merge the two

    # -- run experiment grid --
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
        # cache.clear_exp(uuid)
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)

if __name__ == "__main__":
    main()
