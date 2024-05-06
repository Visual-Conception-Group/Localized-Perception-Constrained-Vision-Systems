import time
import torch
import torchvision
import constants
import models
import data
import patchGD
import pruning
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

def getFLOPS(model, inputs):
    flops = FlopCountAnalysis(model, inputs)
    return flops.total()

# Latency refers to the time it takes for a model 
# to make a prediction
def getLatency(model, inputs):
    start = time.time()
    _ = model(inputs)
    end = time.time()
    latency = end - start
    return latency/constants.batchSize

# Throughput measures the number of predictions 
# a model can make in a given time
def getThroughput(model, inputs):
    model.eval()
    start = time.time()
    _ = model(inputs)
    end = time.time()
    latency = end - start
    return constants.batchSize/latency

def getNumberOfParameters(model):
    return sum(p.numel() for p in model.parameters())

def getModelSize(model1, model2):
    torch.save(model1, constants.basePath+"/models/model183.pth") 
    torch.save(model2, constants.basePath+"/models/model184.pth") 

if __name__=="__main__":
    # model1 = models.getModel1()
    # model1.to(constants.device)
    # model1.load_state_dict(pruning.loadModel("model501"))
    # model2 = models.getModel2()
    # model2.to(constants.device)
    # model2.load_state_dict(pruning.loadModel("model502")) 

    model1 = models.getModel1()
    model1 = nn.DataParallel(model1)
    model1.to(constants.device)
    model2 = models.getModel2()
    model2 = nn.DataParallel(model2)
    model2.to(constants.device)
    model1.load_state_dict(torch.load(constants.basePath+"/final_models/mPatchGD-4096-128-24GB-1.pth"))
    model2.load_state_dict(torch.load(constants.basePath+"/final_models/mPatchGD-4096-128-24GB-2.pth"))
    model1.eval()
    model2.eval()

    resizedImage = None
    for data in data.getDataLoader("train", "PANDA"):
        inputs, categories = data
        inputs.to(constants.device)
        categories.to(constants.device)
        for idx, image in enumerate(inputs): 
            patches, positions = patchGD.sampleKPatches(
                patchGD.getPatches(image))
            resizedImage = torchvision.transforms.Resize(
                    size=(constants.patchSize, constants.patchSize)
                    )(image)
            break
        break
    zMatrix = patchGD.fillZ(inputs, torch.empty((constants.batchSize, 
                    constants.m, constants.n, constants.extractedFeatureLength), 
                    dtype=torch.float32, device=constants.device), model1) 
    # extractedFeatures = model1(patches + resizedImage)

    # print(getFLOPS(model1, patches))  
    # print(getFLOPS(model2, zMatrix))
    print(getLatency(model1, patches))
    print(getLatency(model2, zMatrix))
    # print(getThroughput(model1, patches))
    # print(getThroughput(model2, zMatrix))
    # print(getNumberOfParameters(model1))
    # print(getNumberOfParameters(model2))
    # getModelSize(model1, model2)
    