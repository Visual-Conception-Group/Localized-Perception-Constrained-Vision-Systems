import torch
import models
import constants
from collections import OrderedDict
import torch.nn.utils.prune as prune

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loadModel(model):
    state_dict = torch.load(constants.basePath+"/models/"+model+".pth")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def pruneModel():
    model = models.getModel1()
    model.to(device)
    model.load_state_dict(loadModel("model182"))
    parameters_to_prune = []
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, "weight"))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.4,
    )
    torch.save(model, constants.basePath+"/compressed_models/model182.pth") 

if __name__=="__main__":
    pruneModel()
