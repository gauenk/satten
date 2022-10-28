
# -- python imports --
import os,copy
dcopy = copy.deepcopy
import pprint
import pandas as pd
pp = pprint.PrettyPrinter(indent=4)
from pathlib import Path
from functools import partial

# -- cache --
import cache_io

# -- data --
import data_hub

# -- optical flow --
from satten import flow

# -- package misc --
from satten import utils
from satten.utils.metrics import compute_psnrs,compute_ssims
from satten.configs import compare_fwd as configs

# -- attention packages --
import nat
import n3net
import satten

def run_exp(_cfg):

    # -- init --
    cfg = copy.deepcopy(_cfg)
    cache_io.exp_strings2bools(cfg)
    utils.set_seed(cfg.seed)
    root = (Path(__file__).parents[0] / ".." ).absolute()

    # -- load data --
    data,loaders = data_hub.sets.load(cfg)
    frame_start = optional(cfg,"frame_start",0)
    frame_end = optional(cfg,"frame_end",-1)
    indices = data_hub.filter_subseq(data,cfg.vid_name,frame_start,frame_end)

    # -- iterate over data --
    sample = data.tr[indices[0]]
    clean,noisy = sample['clean'],sample['noisy']

    # -- init timer --
    timer = ExpTimer()

    # -- satten --
    modules = [nat,n3net,satten]
    for module in modules:
        search = getattr(module,'search')
        get_search_config = getattr(module,'extract_search_config')
        search_cfg = get_search_config(cfg)
        with timer(str(module)):
            dists,inds = search(noisy,**search_cfg)

    # -- info --
    print("Running timer.")
    print(timer)

def main():

    # -- init info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_name = "compare_fwd" # current!
    cache = cache_io.ExpCache(".cache_io",cache_name)

    # -- grab default --
    default_cfg = configs.default()

    # -- grid --
    ws,wt,k = [29],[3],[7]
    exp_lists = {"ws":ws,"wt":wt,"k":k}
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
    print(records)

if __name__ == "__main__":
    main()
