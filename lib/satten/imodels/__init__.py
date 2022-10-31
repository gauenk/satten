
# -- misc --
from easydict import EasyDict as edict

# -- our search fxn --
from .imodel import IndexModel

# -- extracting config --
from functools import partial
from ..utils import optional as _optional

# -- auto populate fields to extract config --
_fields = []
def optional_full(init,pydict,field,default):
    if not(field in _fields) and init:
        _fields.append(field)
    return _optional(pydict,field,default)

# -- load model --
def load_imodel(*args,**kwargs):

    # -- allows for all keys to be aggregated at init --
    init = _optional(kwargs,'__init',False) # purposefully weird key
    optional = partial(optional_full,init)

    # -- relevant configs --
    k = optional(kwargs,'k',-1)

    # -- break here if init --
    if init: return

    # -- init model --
    imodel = IndexModel(k=k)
    return imodel

# -- run to populate "_fields" --
load_imodel(__init=True)

def extract_imodel_config(cfg):
    # -- auto populated fields --
    fields = _fields
    _cfg = {}
    for field in fields:
        if field in cfg:
            _cfg[field] = cfg[field]
    return edict(_cfg)

