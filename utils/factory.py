import torch
from models.tacle import TACLE

def get_model(model_name, args): 
    name = model_name.lower()
    if 'tacle' in name:
        return TACLE(args)
    else:
        assert 0
