import torch
import constants
import models
import random
import data
import torch.nn as nn
import pytorch_warmup as warmup

def getPatches(image):
    patches = torch.empty((constants.m, constants.n,
            3, constants.patchSize, constants.patchSize), 
            dtype=torch.float32, device=constants.device)
    i = -1
    for x in range(0, image.shape[1], constants.patchSize):
        i = i+1
        j = -1
        for y in range(0, image.shape[2], constants.patchSize):
            j = j+1
            patches[i][j] = torch.as_tensor(image[:, x:x+constants.patchSize, 
                                  y:y+constants.patchSize], device=constants.device)
    return patches

def sampleKPatches(patches):
    kPatches = torch.empty((constants.k, 3, constants.patchSize, 
                    constants.patchSize), dtype=torch.float32, 
                    device=constants.device)
    positions = []
    currentPositions = {}
    idx = 0
    while idx < constants.k:
        x = random.randint(0, constants.m-1)
        y = random.randint(0, constants.n-1)
        if (x, y) in currentPositions:
            continue
        currentPositions[(x, y)] = None
        kPatches[idx] = patches[x][y]
        positions.append((x, y))
        idx = idx+1
    return kPatches, positions

def fillZ(images, zMatrix, model1):
    for idx, image in enumerate(images):
        patches = getPatches(image)
        patches = torch.reshape(patches, (constants.m * constants.n,
            3, constants.patchSize, constants.patchSize))
        extractedFeatures = model1(patches)
        extractedFeatures = torch.reshape(extractedFeatures,
            (constants.m, constants.n, constants.extractedFeatureLength))
        for i in range(constants.m):
            for j in range(constants.n):
                zMatrix[idx][i][j] = \
                    extractedFeatures[i][j].clone().detach().requires_grad_(True)
    return zMatrix

def getAccuracy(categories, predictions):
    total, correct = 0, 0
    _, outputs = torch.max(predictions.data, 1)
    total += categories.size(0)
    correct += (outputs == categories).sum().item()
    return 100 * correct / total

def train(dataLoader):
    print("Starting training...")
    print("Total mini batches ", len(dataLoader))
    model1 = models.getModel1()
    model1 = nn.DataParallel(model1)
    model1.to(constants.device)
    model2 = models.getModel2()
    model2 = nn.DataParallel(model2)
    model2.to(constants.device)
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=constants.lr)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=constants.lr)
    lr_scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer1)
    warmup_scheduler1 = warmup.UntunedLinearWarmup(optimizer1)
    lr_scheduler2 = torch.optim.lr_scheduler.LinearLR(optimizer2)
    warmup_scheduler2 = warmup.UntunedLinearWarmup(optimizer2)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(constants.epochs):
        accuracy = 0
        torch.cuda.empty_cache()
        for i, data in enumerate(dataLoader):
            if i==len(dataLoader)-1:
                continue
            images, categories = data  
            images = images.to(constants.device)    
            categories = categories.to(constants.device) 
            zMatrix = fillZ(images, torch.empty((constants.batchSize, 
                    constants.m, constants.n, constants.extractedFeatureLength), 
                    dtype=torch.float32, device=constants.device), model1)
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            for j in range(constants.innerIterations):
                for idx, image in enumerate(images): 
                    patches, positions = sampleKPatches(getPatches(image))
                    extractedFeatures = model1(patches)
                    for index, extractedFeature in enumerate(extractedFeatures):
                        x, y = positions[index]
                        zMatrix[idx][x][y] = \
                        extractedFeature.clone().detach().requires_grad_(True)
                    predictions = model2(zMatrix)
                    loss = criterion(predictions, categories)
                    loss.backward(retain_graph=True)
                if (j+1) % constants.accumulationSteps == 0:
                    optimizer1.step() 
                    optimizer2.step()
                    with warmup_scheduler1.dampening():
                        lr_scheduler1.step()
                    with warmup_scheduler2.dampening():
                        lr_scheduler2.step()
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
            predictions = model2(zMatrix)
            accuracy = accuracy + getAccuracy(categories, predictions)
            print("Mini Batch = ", i+1,  " Accuracy = ", getAccuracy(categories, predictions))
        print("Epoch = ", epoch+1, "; Average Accuracy = ", float(accuracy/len(dataLoader)))
    print("Total memory allocated ", torch.cuda.memory_allocated())
    torch.save(model1.state_dict(), constants.basePath+"/models/model503.pth")
    torch.save(model2.state_dict(), constants.basePath+"/models/model504.pth")
    
def test(dataLoader):
    print("Total mini batches ", len(dataLoader))
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
    accuracy = 0
    with torch.no_grad():
        for i, data in enumerate(dataLoader):
            if i==len(dataLoader)-1:
                continue
            images, categories = data
            images = images.to(constants.device)
            categories = categories.to(constants.device) 
            zMatrix = fillZ(images, torch.empty((constants.batchSize, 
                    constants.m, constants.n, constants.extractedFeatureLength), 
                    dtype=torch.float32, device=constants.device), model1)   
            predictions = model2(zMatrix) 
            accuracy = accuracy + getAccuracy(categories, predictions)
            print("Mini Batch = ", i+1,  " Accuracy = ", getAccuracy(categories, predictions))
        print("Average Accuracy = ", float(accuracy/len(dataLoader)))

def transformDataset():
    data.transformDataset("train")
    data.cleanData("train")
    data.transformDataset("validate")
    data.cleanData("validate")
    data.transformDataset("test")
    data.cleanData("test")

def startTraining():
    print("Training the models on the dataset")
    trainDataLoader = data.getDataLoader("train", "PANDA")
    train(trainDataLoader)

def startValidation():
    print("Validating the models on the dataset")
    validateDataLoader = data.getDataLoader("validate", "PANDA")
    test(validateDataLoader)

def startTesting():
    print("Testing the models on the dataset")
    testDataLoader = data.getDataLoader("test", "PANDA")
    test(testDataLoader)

if __name__=="__main__":
    print("GPU Available: ", torch.cuda.is_available())
    startTraining()
