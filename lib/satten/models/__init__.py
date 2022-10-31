
from easydict import EasyDict as edict
import uformer
from ..utils import optional

def load_model(*args,**kwargs):
    model_name = optional(kwargs,"model_name","uformer")
    if model_name == "uformer":
        return uformer.load_model(*args,**kwargs)
    else:
        raise ValueError(f"Uknown model [{model_name}]")

def extract_model_config(cfg):
    model_name = optional(cfg,"model_name","uformer")
    if model_name == "uformer":
        return edict(uformer.extract_model_config(cfg))
    else:
        raise ValueError(f"Uknown model [{model_name}]")


