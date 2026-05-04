from model.loss import *
import torch

from model.vmodel import VModel
from model.dmodel import DModel

def get_loss_fn(name):

    if name == "l2":
        return l2_loss
    elif name == "weighted_l2_loss":
        return weighted_l2_loss
    elif name == "normed_l2":
        return normed_l2_loss
    elif name =="normed_semi_l2":
        return normed_semi_l2_loss
    elif name == "weighted_normed_semi_l2":
        return weighted_normed_semi_l2_loss
    elif name == "l1":
        return l1_loss
    elif name == "normed_l1":
        return normed_l1_loss
    elif name == "normed_semi_l1":
        return normed_semi_l1_loss
    elif name == "cross_l1":
        return cross_l1_loss
    elif name == "normed_cross_l1":
        return normed_cross_l1_loss
    elif name == "cross_l2":
        return cross_l2_loss
    elif name == "normed_cross_l2":
        return normed_cross_l2_loss
    else:
        raise ValueError(f"Unknown loss function: {name}")
    
def get_model(name, path, *args):

    if name == "VModel":
        model = VModel(*args)
    elif name == "DModel":
        model = DModel(*args)
    else:
        raise ValueError(f"Unknown model type: {name}")

    model = model.cuda()
    if name == "VModel":
        if path != "":
            checkpoint = torch.load(path)
            k = checkpoint["k"]
            model.spatial_encoding.update(k)
            model.load_state_dict(checkpoint["model"])
    elif name == "DModel":
        if path != "":
            checkpoint = torch.load(path)
            k = checkpoint["k"]
            model.spatial_encoding.update(k)
            model.load_state_dict(checkpoint["model"])
    else:
        if path != "":
            model.load_state_dict(torch.load(path))

    return model
