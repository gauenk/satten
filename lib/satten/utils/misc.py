
# -- seeding --
import random
import numpy as np
import torch as th


# -- optional --
def optional(pydict,field,default):
    if pydict is None: return default
    if field in pydict: return pydict[field]
    else: return default

# -- seeding --
def set_seed(seed):
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)
