import torchvision
import torch
import constants
import models
import random
import data
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')

def getPatches(image):
    patches = torch.empty((constants.m, constants.n,
            3, constants.patchSize, constants.patchSize), 
            dtype=torch.float32, device=constants.device)
    i = -1
    j = -1
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

def fillZ(imagesPatches, zMatrix, model1):
    for idx, patches in enumerate(imagesPatches):
        patches = torch.reshape(patches, (constants.m * constants.n,
            3, constants.patchSize, constants.patchSize))
        extractedFeatures = model1(patches)
        extractedFeatures = torch.reshape(extractedFeatures,
            (constants.m, constants.n, constants.extractedFeatureLength))
        zMatrix[idx] = \
                    extractedFeatures.clone().detach().requires_grad_(True)
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
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=constants.lr)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=constants.lr)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(constants.epochs):
        accuracy = 0
        torch.cuda.empty_cache()
        for i, data in enumerate(dataLoader):
            if i==len(dataLoader)-1:
                continue
            images, categories = data 
            images = images.to(constants.device)   
            categories = categories.type(torch.LongTensor) 
            categories = categories.to(constants.device)      
            imagesPatches = []
            for image in images:
                imagesPatches.append(getPatches(image))
            zMatrix = fillZ(imagesPatches, torch.empty((constants.batchSize, 
                    constants.m, constants.n, constants.extractedFeatureLength), 
                    dtype=torch.float32, device=constants.device), model1)
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            for j in range(constants.innerIterations):
                resizedImages = torch.empty((constants.batchSize,
                    3, constants.patchSize, constants.patchSize), 
                    dtype=torch.float32, device=constants.device)
                for idx, image in enumerate(images):
                    resizedImages[idx] = torchvision.transforms.Resize(
                        size=(constants.patchSize, constants.patchSize)
                        )(image).to(constants.device) 
                # Extracting features of resized images
                resizedImagesExtractedFeatures = model1(resizedImages)
                resizedImagesExtractedFeaturesRepeated = \
                    torch.empty((constants.batchSize, 
                    constants.n*constants.extractedFeatureLength), 
                    dtype=torch.float32, device=constants.device)
                for idx in range(constants.batchSize):
                    resizedImagesExtractedFeaturesRepeated[idx] = \
                        resizedImagesExtractedFeatures[idx].repeat(constants.n)
                resizedImagesExtractedFeaturesRepeated = \
                    torch.reshape(resizedImagesExtractedFeaturesRepeated, \
                    (constants.batchSize, 1, constants.n,
                     constants.extractedFeatureLength))
                for idx, image in enumerate(images): 
                    patches, positions = sampleKPatches(imagesPatches[idx])
                    extractedFeatures = model1(patches)
                    for index, extractedFeature in enumerate(extractedFeatures):
                        x, y = positions[index]
                        zMatrix[idx][x][y] = \
                        extractedFeature.detach().clone().requires_grad_(True)
                    temp = torch.cat([zMatrix, \
                            resizedImagesExtractedFeaturesRepeated], dim=1)
                    predictions = model2(temp)
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
            currentAccuracy = getAccuracy(categories, predictions)
            accuracy = accuracy + currentAccuracy
            print("Epoch = ", epoch+1, " Mini Batch = ", i+1,  " Accuracy = ", currentAccuracy)
        print("Epoch = ", epoch+1, "; Average Accuracy = ", float(accuracy/len(dataLoader)))
    print("Total memory allocated ", torch.cuda.memory_allocated())
    torch.save(model1.state_dict(), constants.basePath+"/models/mPatchGD-4096-128-24GB-1.pth")
    torch.save(model2.state_dict(), constants.basePath+"/models/mPatchGD-4096-128-24GB-2.pth")
    
def test(dataLoader):
    print("Total mini batches ", len(dataLoader))
    model1 = models.getModel1()
    model1 = nn.DataParallel(model1)
    model1.to(constants.device)
    model2 = models.getModel2()
    model2 = nn.DataParallel(model2)
    model2.to(constants.device)
    model1.load_state_dict(torch.load(constants.basePath+"/models/model183.pth"))
    model2.load_state_dict(torch.load(constants.basePath+"/models/model184.pth"))
    model1.eval()
    model2.eval()
    accuracy = 0
    with torch.no_grad():
        for i, data in enumerate(dataLoader):
            if i==len(dataLoader)-1:
                continue
            images, categories = data
            images = images.to(constants.device)
            imagesPatches = []
            for image in images:
                imagesPatches.append(getPatches(image))
            categories = categories.to(constants.device) 
            zMatrix = fillZ(imagesPatches, torch.empty((constants.batchSize, 
                    constants.m, constants.n, constants.extractedFeatureLength), 
                    dtype=torch.float32, device=constants.device), model1)
            resizedImageFeatures = model1(torchvision.transforms.Resize(
                size=(constants.patchSize, constants.patchSize)
                 )(images)).view((constants.batchSize,
                1, 1, constants.extractedFeatureLength))   
            predictions = model2(zMatrix + resizedImageFeatures) 
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
