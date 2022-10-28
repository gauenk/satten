

# -- extracting config --
from functools import partial
from ..utils import optional as _optional

# -- auto populate fields to extract config --
_fields = []
def optional_full(init,pydict,field,default):
    if not(field in _fields) and init:
        _fields.append(field)
    return _optional(pydict,field,default)

# -- init search --
def init_search(*args,**kwargs):

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
    ws = optional(kwargs,'ws',8)
    wt = optional(kwargs,'wt',0)
    nbwd = optional(kwargs,'nbwd',1)
    rbwd = optional(kwargs,'rbwd',False)
    exact = optional(kwargs,'exact',False)
    bs = optional(kwargs,'bs',-1)
    dil = optional(kwargs,'dilation',1)
    chnls = optional(kwargs,'chnls',-1)
    name = optional(kwargs,'sfxn','l2')

    # -- break here if init --
    if init: return

    # -- init model --
    search = dnls.init("%s_with_heads" % name, None, None,
                       k=k, ps=ps, pt=pt, ws=ws, wt=wt, nheads=nheads,
                       chnls=chnls,dilation=dil, stride0=stride0,
                       stride1=stride1, use_k=True, use_adj=True,
                       use_k=True, use_adj=True, reflect_bounds=True,
                       search_abs=False, full_ws=False, nbwd=nbwd, exact=exact,
                       h0_off=0,w0_off=0,h1_off=0,w1_off=0,remove_self=False,
                       anchor_self=False,rbwd=rbwd)
    return search


# -- run to populate "_fields" --
get_search(__init=True)

def extract_search_config(cfg):
    # -- auto populated fields --
    fields = _fields
    _cfg = {}
    for field in fields:
        if field in cfg:
            _cfg[field] = cfg[field]
    return _cfg
