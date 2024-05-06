import torch
import torchvision
import constants
import fcos
import patchGD
import cv2
import numpy
from matplotlib import pyplot as plt
from torchmetrics.detection import MeanAveragePrecision

N = constants.m * constants.n
cnn1 = torch.nn.Conv2d(constants.H[0], constants.W[0], 1).to(constants.device)
cnn2 = torch.nn.Conv2d(constants.H[1], constants.W[1], 1).to(constants.device)
cnn3 = torch.nn.Conv2d(constants.H[2], constants.W[2], 1).to(constants.device)
cnn4 = torch.nn.Conv2d(constants.H[3], constants.W[3], 1).to(constants.device)
cnn5 = torch.nn.Conv2d(constants.H[4], constants.W[4], 1).to(constants.device)
cnns = [cnn1, cnn2, cnn3, cnn4, cnn5]
maxPool = torch.nn.MaxPool1d(N+1)
maxPool.to(constants.device)
optimizerCNN = torch.optim.SGD([*cnn1.parameters(),
        *cnn2.parameters(), *cnn3.parameters(), *cnn4.parameters(),
        *cnn5.parameters()], lr=constants.lr)

def drawGroundTruthBoundingBoxes():
    color = (0,255,0)
    imageIndex = 3
    images = torch.load(constants.basePath + "/dhd_traffic/dhdImages4096Train.pt")
    targets = torch.load(constants.basePath +  "/dhd_traffic/dhdTargets4096Train.pt")
    transform = torchvision.transforms.ToPILImage()
    inferenceResults = []
    for index in range(len(targets[imageIndex]['boxes'])):
        inferenceResults.append(
            { "left": targets[imageIndex]["boxes"][index][0].item(),
            "top": targets[imageIndex]["boxes"][index][3].item(),
            "right": targets[imageIndex]["boxes"][index][2].item(),
            "bottom": targets[imageIndex]["boxes"][index][1].item(), 
            "label": targets[imageIndex]["labels"][index].item() }
        )
    
    imageData = cv2.cvtColor(numpy.array(transform(images[imageIndex])), 
                            cv2.COLOR_RGB2BGR)
    for i,res in enumerate(inferenceResults):
        # print(res)
        left = int(res['left'])
        top = int(res['top'])
        right = int(res['right'])
        bottom = int(res['bottom'])
        label = str(res['label'])
        imgHeight, imgWidth, _ = imageData.shape
        thick = int((imgHeight + imgWidth) // 900)
        cv2.rectangle(imageData,(left, top), (right, bottom), color, thick)
        cv2.putText(imageData, label, (left, top), 0, 1e-3 * imgHeight, color, thick//3)
    cv2.imwrite(constants.basePath + "/mergedImageCoco.png", imageData)

def drawBoundingBoxes():
    model = fcos.fcos_resnet50_fpn() 
    model.load_state_dict(torch.load(constants.basePath+"/models/detection1.pth"))
    model.to(constants.device)
    model.eval()
    color = (0,255,0)
    imageIndex = 7
    images = torch.load(constants.basePath + "/coco2017/cocoMergedImagesTestNormalized.pt")

    transform = torchvision.transforms.ToPILImage()
    predictions = model(images[imageIndex].unsqueeze(0), 
                        getZMatrix(images[imageIndex].unsqueeze(0), model))
    inferenceResults = []
    for index in range(len(predictions[0]['boxes'])):
        inferenceResults.append(
            { "left": predictions[0]["boxes"][index][0].item(),
            "top": predictions[0]["boxes"][index][3].item(),
            "right": predictions[0]["boxes"][index][2].item(),
            "bottom": predictions[0]["boxes"][index][1].item(), 
            "label": predictions[0]["labels"][index].item() }
        )
    
    images = torch.load(constants.basePath + "/coco2017/cocoMergedImagesTest.pt")
    imageData = cv2.cvtColor(numpy.array(transform(images[imageIndex])), 
                            cv2.COLOR_RGB2BGR)
    for res in inferenceResults:
        print(res)
        left = int(res['left'])
        top = int(res['top'])
        right = int(res['right'])
        bottom = int(res['bottom'])
        label = str(res['label'])
        imgHeight, imgWidth, _ = imageData.shape
        thick = int((imgHeight + imgWidth) // 900)
        cv2.rectangle(imageData,(left, top), (right, bottom), color, thick)
        cv2.putText(imageData, label, (left, top), 0, 1e-3 * imgHeight, color, thick//3)
    cv2.imwrite(constants.basePath + "/bboxesPlotted.png", imageData)

def getZMatrix(imagesBatch, model):
    zMatrix = [torch.empty((constants.batchSize, 256, constants.H[0], constants.W[0]),
                        dtype=torch.float32, device=constants.device), 
                   torch.empty((constants.batchSize, 256, constants.H[1], constants.W[1]), 
                        dtype=torch.float32, device=constants.device), 
                   torch.empty((constants.batchSize, 256, constants.H[2], constants.W[2]), 
                        dtype=torch.float32, device=constants.device), 
                   torch.empty((constants.batchSize, 256, constants.H[3], constants.W[3]), 
                        dtype=torch.float32, device=constants.device),
                   torch.empty((constants.batchSize, 256, constants.H[4], constants.W[4]), 
                        dtype=torch.float32, device=constants.device)]
     
    for batchIdx, image in enumerate(imagesBatch):
        imageAsPatch = torchvision.transforms.Resize(
                    size=(constants.patchSize, constants.patchSize)
                    )(image).to(constants.device) 
        imageAsPatch = torch.reshape(imageAsPatch, (1, 3, 
                constants.patchSize, constants.patchSize))
        patches = torch.reshape(patchGD.getPatches(image), (N, 3, 
                constants.patchSize, constants.patchSize))        
        patches = torch.cat([patches, imageAsPatch], dim=0)
        patches = patches.to(constants.device) 

        # model.get_features gives output features of fcos FPN
        extractedFeatures = model.get_features(patches)

        # filling zMatrix with output features of fcos FPN (dim as N X 256 X H X W)
        for index, extractedFeature in enumerate(extractedFeatures):
            _,F,H,W = extractedFeature.size()
            extractedFeature = torch.reshape(extractedFeature, (H,
                W, F*(N+1)))  
            extractedFeature = cnns[index](extractedFeature)
            extractedFeature = cnns[index](extractedFeature)
            extractedFeature = maxPool(extractedFeature)
            extractedFeature = torch.reshape(extractedFeature, (256, H,
                W))   
            zMatrix[index][batchIdx] = \
                extractedFeature.clone().detach().requires_grad_(False)
    return zMatrix

def train():
    print("Starting training...")
    images = torch.load(constants.basePath + "/dhd_traffic/dhdImages4096Train.pt")
    # targets is a list of dictionaries with keys as 'boxes' and 'labels'
    targets = torch.load(constants.basePath + "/dhd_traffic/dhdTargets4096Train.pt")
    print("Total Mini Batches: ", len(images)//constants.batchSize)
    model = fcos.fcos_resnet50_fpn(num_classes=constants.numClasses)
    model.to(constants.device)
    optimizer =  torch.optim.SGD(model.parameters(), lr=constants.lr)
    trainingRegressionLosses = []
    trainingClassificationLosses = []
    for epoch in range(constants.epochs):
        for idx in range(0, len(images), constants.batchSize):
            miniBatch = idx//constants.batchSize
            imagesBatch = images[idx:idx+constants.batchSize]
            targetsBatch = targets[idx:idx+constants.batchSize]
            if len(imagesBatch)<constants.batchSize:
                break

            # moving tensors in targetsBatch to constants.device
            for i, target in enumerate(targetsBatch):
                targetsBatch[i] = {
                    'boxes': target['boxes'].to(constants.device),
                    'labels': target['labels'].to(constants.device)
                }

            optimizer.zero_grad() 
            optimizerCNN.zero_grad()

            # Original images are passed to FCOS model for calculating anchors
            # which are used in the postprocessing of detections given by FCOS model.
            # If we have a set of pre-defined anchors (in dataset), we can pass those 
            # via small modifications to fcos.py and remove original images from method below.
            losses = model(imagesBatch, getZMatrix(imagesBatch, model), targetsBatch)
            print("Epoch: ", epoch+1, "Mini Batch: ", str(miniBatch+1), "Losses: ", losses)
            
            losses['classification'].backward(retain_graph=True)
            losses['bbox_regression'].backward(retain_graph=True)

            optimizer.step()
            optimizer.zero_grad()
            optimizerCNN.step()
            optimizerCNN.zero_grad()

        trainingClassificationLosses.append(losses['classification'].item())
        trainingRegressionLosses.append(losses['bbox_regression'].item())
        
    plt.plot(trainingRegressionLosses, label='IOU Loss')  
    plt.plot(trainingClassificationLosses, label='Focal loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")  
    plt.legend(loc="upper right")
    plt.savefig('trainingLossCurve1.png')

    torch.save(model.state_dict(), constants.basePath+"/models/detection4.pth")
    
def test():
    images = torch.load(constants.basePath + "/coco2017/cocoImagesTest.pt")
    # targets is a list of dictionaries with keys as 'boxes' and 'labels'
    targets = torch.load(constants.basePath + "/coco2017/cocoTargetsTest.pt")  
    model = fcos.fcos_resnet50_fpn(num_classes=constants.numClasses) 
    model.load_state_dict(torch.load(constants.basePath+"/models/detection2.pth"))
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.75])
    model.to(constants.device)
    model.eval()
    print("Total Mini Batches: ", len(images)//constants.batchSize)
    for idx in range(0, len(images), constants.batchSize):
        miniBatch = idx//constants.batchSize
        imagesBatch = images[idx:idx+constants.batchSize]
        targetsBatch = targets[idx:idx+constants.batchSize]
        if len(imagesBatch) < constants.batchSize:
            break

        # moving tensors in targetsBatch to constants.device
        for i, target in enumerate(targetsBatch):
            targetsBatch[i] = {
                'boxes': target['boxes'].to(constants.device),
                'labels': target['labels'].to(constants.device)
            }
                
        predictions = model(imagesBatch, getZMatrix(imagesBatch, model)) 
        metric.update(predictions, targetsBatch)
        print("Mini Batch = ", miniBatch+1,  "AP = ", metric.compute()['map'].item())
        
    print("Mean AP = ", metric.compute()['map'].item())

def startTraining():
    print("Training the models on the dataset")
    train()

def startValidation():
    print("Validating the models on the dataset")
    test()

def startTesting():
    print("Testing the models on the dataset")
    test()

if __name__=="__main__":
    print("GPU Available: ", torch.cuda.is_available())
    train()
