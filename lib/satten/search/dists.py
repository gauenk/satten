
# -- misc --
from easydict import EasyDict as edict
from einops import rearrange,repeat

# -- our compute_dists fxn --
import dnls

# -- extracting config --
from functools import partial
from ..utils import optional as _optional

# -- auto populate fields to extract config --
_fields = []
def optional_full(init,pydict,field,default):
    if not(field in _fields) and init:
        _fields.append(field)
    return _optional(pydict,field,default)

# -- init dists --
def init_dists(*args,**kwargs):

    # -- allows for all keys to be aggregated at init --
    init = _optional(kwargs,'__init',False) # purposefully weird key
    optional = partial(optional_full,init)

    # -- relevant configs --
    k = optional(kwargs,'k',-1)
    ps = optional(kwargs,'ps',1)
    pt = optional(kwargs,'pt',1)
    nheads = optional(kwargs,'nheads',1)
    stride0 = optional(kwargs,'stride0',1)
    stride1 = optional(kwargs,'stride1',1)
    nbwd = optional(kwargs,'nbwd',1)
    rbwd = optional(kwargs,'rbwd',False)
    exact = optional(kwargs,'exact',False)
    dil = optional(kwargs,'dilation',1)
    chnls = optional(kwargs,'chnls',-1)
    name = optional(kwargs,'sfxn','prod')
    use_adj = False
    anchor_self = optional(kwargs,'dists_anchor_self',False)
    use_k = k > 0
    reflect_bounds = False

    # -- break here if init --
    if init: return

    # -- init model --
    prod_dists = dnls.search.init("prod_dists", k, ps, pt, nheads,
                                  chnls=-1,dilation=dil,
                                  stride0=stride0, stride1=stride1,
                                  reflect_bounds=reflect_bounds,use_k=use_k,
                                  search_abs=False,use_adj=use_adj,
                                  anchor_self=anchor_self,
                                  exact=exact)
    return prod_dists


# -- run to populate "_fields" --
init_dists(__init=True)

def extract_dists_config(cfg):
    # -- auto populated fields --
    fields = _fields
    _cfg = {}
    for field in fields:
        if field in cfg:
            _cfg[field] = cfg[field]
    return edict(_cfg)


# -- compute dists --
def run_dists(vid0,inds,cfg):

    # -- unpack --
    K_s = _optional(cfg,'stream_k',10)
    nheads = _optional(cfg,'nheads',1)
    output_as_vid_shape = _optional(cfg,'output_as_vid_shape',True)
    B,T,C,H,W = vid0.shape

    # -- init --
    vid1 = _optional(cfg,'vid1',vid0)
    prod_dists = init_dists(**cfg)
    inds = inds.view(B,nheads,-1,K_s,3)
    dists,inds = prod_dists(vid0,inds,0,vid1)

    # -- match output shape --
    if output_as_vid_shape:

        # -- compute nums --
        B,T,C,H,W = vid0.shape
        stride0 = prod_dists.stride0
        nH = (H-1)//stride0+1
        nW = (W-1)//stride0+1

        # -- reshape --
        dists = rearrange(dists,'b H (t h w) k -> b H t h w k',h=nH,w=nW)
        inds = rearrange(inds,'b H (t h w) k tr -> b H t h w k tr',h=nH,w=nW)

    return dists,inds
