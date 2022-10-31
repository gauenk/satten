from easydict import EasyDict as edict

def default():

    cfg = edict()

    # -- general --
    cfg.device = "cuda:0"
    cfg.seed = 123

    # -- data --
    cfg.dname = "davis"
    cfg.dset = "tr"
    cfg.sigma = 30.
    cfg.vid_name = "sheep"
    cfg.nframes = 3
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start+cfg.nframes-1
    cfg.frame_end = -1 if cfg.nframes == 0 else cfg.frame_end

    # -- streaming --
    cfg.memory_nframes = 5
    cfg.memory_k = 15
    cfg.stream_nframes = 4
    cfg.stream_k = 7

    # -- search --
    cfg.stride0 = 4
    cfg.stride1 = 1

    # -- pick model --
    cfg.model_name = "uformer"

    # -- uformer --
    cfg.model = edict()
    cfg.model.pretrained_path = "/home/gauenk/Documents/packages/uformer/output/checkpoints/53e635ce-ec14-4e1f-a80e-5d2019d4dbae-epoch=28.ckpt"
    cfg.model.load_pretrained = True
    cfg.model.pretrained_prefix = "net."
    cfg.model.num_heads = '1-2-4-8-16'
    cfg.model.in_attn_mode = "pd-pd-pd-pd-pd"
    cfg.model.attn_mode = "pd-pd-pd-pd-pd"
    cfg.model.attn_reset = False
    cfg.model.embed_dim = 32
    cfg.model.stride0 = '4-4-2-1-1'
    cfg.model.stride1 = '1-1-1-1-1'
    cfg.model.ws = "29-15-9-9-9"
    cfg.model.wt = 0
    cfg.model.k = 7
    cfg.model.ps = '7-7-5-3-3'
    cfg.model.model_depth = "1-2-8-8-2"
    cfg.model.input_proj_depth = 1
    cfg.model.output_proj_depth = 1
    cfg.model.qk_frac = 0.25


    return cfg
