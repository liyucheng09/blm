from . inst import InsTLM
from . blm import BLM
from . lblm import LBLM
from .blm_pretrained import PBLM


def get_model_class(model_type):
    if model_type == 'blm':
        return BLM
    elif model_type == 'inst':
        return InsTLM
    elif model_type == 'lblm':
        return LBLM
    elif model_type == 'pblm':
        return PBLM
    else:
        raise ValueError('Unknown model ' + model_type)
