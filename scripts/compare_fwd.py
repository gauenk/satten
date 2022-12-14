
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

# -- cache --
import cache_io

# -- data --
import data_hub

# -- optical flow --
from satten import flow

# -- timer --
from satten.utils.timer import ExpTimer,TimeIt

# -- package misc --
from satten import utils
from satten.utils import optional
from satten.configs import compare_fwd as configs
from satten.utils.metrics import compute_psnrs,compute_ssims

# -- attention packages --
import nat
import n3net
import satten
import uformer

def run_exp(_cfg):

    # -- init --
    cfg = copy.deepcopy(_cfg)
    cache_io.exp_strings2bools(cfg)
    utils.set_seed(cfg.seed)
    root = (Path(__file__).parents[0] / ".." ).absolute()
    device = cfg.device

    # -- load data --
    data,loaders = data_hub.sets.load(cfg)
    frame_start = optional(cfg,"frame_start",0)
    frame_end = optional(cfg,"frame_end",-1)
    indices = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,frame_start,frame_end)

    # -- iterate over data --
    sample = data[cfg.dset][indices[0]]
    clean,noisy = sample['clean'][None,:],sample['noisy'][None,:]
    noisy = repeat(noisy,'b t c h w -> b t (r c) h w',r=cfg.creps)
    clean,noisy = clean.to(device),noisy.to(device)
    B,T,C,H,W = clean.shape

    # -- init timer --
    timer = ExpTimer()

    # -- satten --
    modules = {"satten":satten,"n3net":n3net,"nat":nat,"uformer":uformer}
    for module_name,module in modules.items():
        init_search = getattr(module,'init_search')
        get_search_config = getattr(module,'extract_search_config')
        cfg.use_tiled = "tiled" in module_name
        search_cfg = get_search_config(cfg)
        search = init_search(**search_cfg)
        ntotal = T * ((H-1)//search.stride0+1) * ((W-1)//search.stride0+1)
        for rep in range(cfg.nreps):
            if rep == 0:
                dists,inds = search(noisy,0,ntotal)
            else:
                with TimeIt(timer,module_name + "_%d" % rep):
                    dists,inds = search(noisy,0,ntotal)

    # -- info --
    print("Running timer.")
    print(timer)

    # -- results --
    results = edict()
    for name,time in timer.items():
        print(name)
        results[name] = time
    print(results)

    return results

def append_ave_std(records):
    module_names = ["satten","n3net","nat","uformer"]
    for mname in module_names:
        df = records.filter(like=mname)
        if df.empty: break
        field = "ave_%s"%mname
        records[field] = df.mean(1)
        field = "std_%s"%mname
        records[field] = df.std(1)
    return records

def main():

    # -- init info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_name = "compare_fwd" # current!
    cache = cache_io.ExpCache(".cache_io",cache_name)
    # cache.clear()

    # -- grab default --
    default_cfg = configs.default()
    # default_cfg.isize = "100_400"
    # default_cfg.isize = "400_400"
    # default_cfg.isize = "423_454"
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
    for isize,sdf in records.groupby("isize"):
        for creps,cdf in sdf.groupby("creps"):
            df = cdf
            df = append_ave_std(df)
            summary = df.filter(like='ave')
            summary_std = df.filter(like='std')
            print("-=-=-=-=- Summary -=-=-=-=-")
            print(" (creps: %d, isize: %s) " % (creps,isize) )
            print(summary)
            print(summary_std)

if __name__ == "__main__":
    main()
