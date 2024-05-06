import torchvision
import models
import torch
import data
import constants
import oursPatchGDv1
import warnings
import torch.nn as nn

warnings.filterwarnings('ignore')

def fineTune(dataLoader):
    print("Starting fine-tuning on edge device...")
    print("Total mini batches ", len(dataLoader))
    model1 = models.getModel1()
    model2 = models.getModel2()
    model1.load_state_dict(torch.load(constants.basePath+"/models/model183.pth"))
    model2.load_state_dict(torch.load(constants.basePath+"/models/model184.pth"))
    model2.fc = nn.Linear(512, 28)
    model1.to(constants.device)
    model2.to(constants.device)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=constants.lr)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=constants.lr)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(constants.epochs):
        accuracy = 0
        torch.cuda.empty_cache()
        for i, data in enumerate(dataLoader):
            if i==len(dataLoader)-1:
                break
            images, categories = data  
            images = images.to(constants.device)    
            categories = categories.to(constants.device) 
            imagesPatches = []
            for image in images:
                imagesPatches.append(oursPatchGDv1.getPatches(image))
            zMatrix = oursPatchGDv1.fillZ(imagesPatches, torch.empty((constants.batchSize, 
                    constants.m, constants.n, constants.extractedFeatureLength), 
                    dtype=torch.float32, device=constants.device), model1)
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            for j in range(constants.innerIterations):
                for idx, image in enumerate(images): 
                    patches, positions = oursPatchGDv1.sampleKPatches(imagesPatches[idx])
                    resizedImage = torchvision.transforms.Resize(
                            size=(constants.patchSize, constants.patchSize)
                            )(image)
                    extractedFeatures = model1(patches + resizedImage)
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
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
            imageFeatures = model1(torchvision.transforms.Resize(
                size=(constants.patchSize, constants.patchSize)
                 )(images)).view((constants.batchSize,
                1, 1, constants.extractedFeatureLength)) 
            predictions = model2(zMatrix + imageFeatures)
            currentAccuracy = oursPatchGDv1.getAccuracy(categories, predictions)
            accuracy = accuracy + currentAccuracy
            print("Epoch = ", epoch+1, " Mini Batch = ", i+1,  " Accuracy = ", currentAccuracy)
        print("Epoch = ", epoch+1, "; Average Accuracy = ", float(accuracy/len(dataLoader)))
    print("Total memory allocated ", torch.cuda.memory_allocated())
    torch.save(model1.state_dict(), constants.basePath+"/final_models/model183.pth")
    torch.save(model2.state_dict(), constants.basePath+"/final_models/model184.pth")

def train(dataLoader):
    print("Total mini batches ", len(dataLoader))
    model1 = models.getModel1()
    model2 = models.getModel2()
    model1.to(constants.device)
    model2.to(constants.device)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=constants.lr)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=constants.lr)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(constants.epochs):
        accuracy = 0
        torch.cuda.empty_cache()
        for i, data in enumerate(dataLoader):
            if i==len(dataLoader)-1:
                break
            images, categories = data  
            images = images.to(constants.device)    
            categories = categories.to(constants.device) 
            imagesPatches = []
            for image in images:
                imagesPatches.append(oursPatchGDv1.getPatches(image))
            zMatrix = oursPatchGDv1.fillZ(imagesPatches, torch.empty((constants.batchSize, 
                    constants.m, constants.n, constants.extractedFeatureLength), 
                    dtype=torch.float32, device=constants.device), model1)
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            for j in range(constants.innerIterations):
                for idx, image in enumerate(images): 
                    patches, positions = oursPatchGDv1.sampleKPatches(imagesPatches[idx])
                    resizedImage = torchvision.transforms.Resize(
                            size=(constants.patchSize, constants.patchSize)
                            )(image)
                    extractedFeatures = model1(patches + resizedImage)
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
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
            imageFeatures = model1(torchvision.transforms.Resize(
                size=(constants.patchSize, constants.patchSize)
                 )(images)).view((constants.batchSize,
                1, 1, constants.extractedFeatureLength)) 
            predictions = model2(zMatrix + imageFeatures)
            currentAccuracy = oursPatchGDv1.getAccuracy(categories, predictions)
            accuracy = accuracy + currentAccuracy
            print("Epoch = ", epoch+1, " Mini Batch = ", i+1,  " Accuracy = ", currentAccuracy)
        print("Epoch = ", epoch+1, "; Average Accuracy = ", float(accuracy/len(dataLoader)))
    print("Total memory allocated ", torch.cuda.memory_allocated())
    torch.save(model1.state_dict(), constants.basePath+"/final_models/mPatchGD-512-256-4GB-1.pth")
    torch.save(model2.state_dict(), constants.basePath+"/final_models/mPatchGD-512-256-4GB-2.pth")
    
if __name__=="__main__":
    print("GPU Available: ", torch.cuda.is_available())
    dataLoader = data.getDataLoader("train", "UltraMNIST")
    train(dataLoader)
    