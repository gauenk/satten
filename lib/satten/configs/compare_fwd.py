from easydict import EasyDict as edict

def default():
    cfg = edict()
    cfg.device = "cuda:0"
    cfg.seed = 123
    cfg.dname = "davis"
    cfg.dset = "tr"
    cfg.sigma = 30.
    cfg.vid_name = "sheep"
    cfg.nframes = 3
    cfg.frame_start = 0
    cfg.stride0 = 4
    cfg.frame_end = cfg.frame_start+cfg.nframes-1
    cfg.frame_end = -1 if cfg.nframes == 0 else cfg.frame_end
    return cfg
