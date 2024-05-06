import shutil
import csv
import os
import torchvision
import zipfile
import torch
import constants
import PIL
import json
import math
from PIL import Image

def getDataLoader(type, dataset): # type defines the data as train, validate or test 
    if dataset == "PANDA":
        imageSize = constants.pandaImageSize
    else:
        imageSize = constants.aidImageSize
    print("Converting dataset to a pytorch dataloader object for " + type)
    dataTransform = torchvision.transforms.Compose([
        torchvision.transforms.Pad(30, 255), # applying white border on image
        torchvision.transforms.Resize(
            size=(imageSize, imageSize)
        ),
        torchvision.transforms.ToTensor(),   
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    transformedDataset = torchvision.datasets.ImageFolder(
        root=constants.basePath+'/'+dataset+'/' + type + "/",
        transform=dataTransform)
    dataLoader = torch.utils.data.DataLoader(transformedDataset,
        batch_size=constants.batchSize, shuffle=True, num_workers=1)
    print("Successfully converted dataset to a pytorch dataloader object!")
    return dataLoader

def transformDataset(type): # type defines the data as train, validate or test 
    file = type + ".csv"
    print("Transforming the dataset for " + type)
    with open(constants.basePath + "/dataset/"+ file, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            imageLabel = str(row[0])
            if imageLabel == "id":
                continue
            category = str(row[1])
            # try:
            if not os.path.exists(
                constants.basePath+"/UltraMNIST/" + 
                    type + "/" + category):
                os.mkdir(
                    constants.basePath+"/UltraMNIST/" 
                    + type + "/" + category)
            shutil.move(constants.basePath+"/dataset/" + "train/" + imageLabel + 
                    ".jpeg", constants.basePath+"/UltraMNIST/" 
                    + type + "/" + category)
            # except:
                # print("Failed to extract image " + imageLabel + ".tiff")
    print("Successfully transformed the dataset!")

# cleanData removes corrupt images from dataset
def cleanData(type, dataset):
    folder_paths = [
        constants.basePath+"/"+dataset+"/"+type+'/0',
        constants.basePath+"/"+dataset+"/"+type+'/1',
        constants.basePath+"/"+dataset+"/"+type+'/2',
        constants.basePath+"/"+dataset+"/"+type+'/3',
        constants.basePath+"/"+dataset+"/"+type+'/4',
        constants.basePath+"/"+dataset+"/"+type+'/5',
    ]
    for folder_path in folder_paths:
        for filename in os.listdir(folder_path):
            try:
                _ = Image.open(os.path.join(folder_path, filename))
            except PIL.UnidentifiedImageError as e:
                os.remove(os.path.join(folder_path, filename))

def extractDataset(datasetPath, folderName):
    with zipfile.ZipFile(
        datasetPath, 'r') as zip_ref:
         zip_ref.extractall(constants.basePath+"/"+folderName)

# Below method is for processing COCO dataset
# and convert the dataset to the format needed by the detection model.
def saveImagesBboxesAndLabels(dataset, type):
    images=[]
    targets=[]
    imageToObj={}
    imageToBBoxes={}
    imageToLabels={}
    transform = torchvision.transforms.Compose([ 
                    torchvision.transforms.ToTensor(),
                    # torchvision.transforms.Resize(
                    #     size=(constants.cocoImageSize, constants.cocoImageSize)),  
                    # torchvision.transforms.Normalize(
                    #     mean=[0.485, 0.456, 0.406],
                    #     std=[0.229, 0.224, 0.225]
                    # )
                ])
    if dataset=="coco":
        file = open(constants.basePath + 
                    '/coco2017/annotations/instances_train2017.json')
        data = json.loads(file.read())
        for idx, annotation in enumerate(data['annotations']):
            imageID = str(annotation['image_id'])
            while len(imageID)<12:
                imageID = "0"+imageID
            if imageID not in imageToObj:
                image = Image.open(constants.basePath + 
                            "/coco2017/" + type + "2017/" + imageID + ".jpg")
                if image.mode == "L":
                    continue
                imageToObj[imageID] = image
            if imageID not in imageToBBoxes:
                imageToBBoxes[imageID] = []
            if imageID not in imageToLabels:
                imageToLabels[imageID] = []
            imageToBBoxes[imageID].append(annotation['bbox'])
            imageToLabels[imageID].append(annotation['category_id'])
        currentImgs = []
        currentBboxes = []
        currentLabels = []
        # Following code adds a set of images to a high resolution white canvas
        # to be able to perform detection on large sized image
        for imageID, imageObj in imageToObj.items():
            if len(currentImgs) == constants.numberOfImagesToMerge:
                newImage = Image.new('RGB',(constants.newImageSize, constants.newImageSize),
                    (250,250,250))
                newBboxes = []
                newLabels = []
                positions = []
                positionIdx = 0
                w,h = 1,1
                for imageIdx in range(constants.numberOfImagesToMerge):
                    rowIdx = imageIdx//constants.numberOfImagesInOneRow
                    colIdx = imageIdx%constants.numberOfImagesInOneRow
                    positions.append([colIdx*w, rowIdx*h])
                    w = max(w,currentImgs[imageIdx].size[0])
                    h = max(h,currentImgs[imageIdx].size[1])
                while len(currentImgs)>0: 
                    currentImg = currentImgs.pop()
                    x = positions[positionIdx][0]
                    y = positions[positionIdx][1]
                    newImage.paste(currentImg, (x,y))
                    bboxes = currentBboxes.pop()
                    for bbox in bboxes:
                        xmin = bbox[0]+x
                        ymin = bbox[1]+y
                        xmax = (bbox[0]+bbox[2])+x
                        ymax = (bbox[1]+bbox[3])+y
                        newBboxes.append([xmin, ymin, xmax, ymax])
                    newLabels.extend(currentLabels.pop())
                    positionIdx = positionIdx+1
                images.append(transform(newImage))
                targets.append({
                    'boxes': torch.as_tensor(newBboxes),
                    'labels': torch.as_tensor(newLabels)
                    })
                currentImgs = []
                currentBboxes = []
                currentLabels = []
            currentImgs.append(imageObj)
            currentBboxes.append(imageToBBoxes[imageID])
            currentLabels.append(imageToLabels[imageID])
    file1 = "cocoMergedImages4096"
    file2 = "cocoTargetsForMergedImages4096"
    torch.save(images, constants.basePath + "/coco2017/"+file1+".pt")
    torch.save(targets, constants.basePath + "/coco2017/"+file2+".pt")

def transformSODADDataset():
    imageIDToImageObj = {}
    imageIDToBboxes = {}
    imageIDToTargets = {}
    file = open(constants.basePath + 
        '/SODA-D/Annotations/train.json')
    data = json.loads(file.read())
    transform = torchvision.transforms.Compose([ 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(
                size=(constants.newImageSize, constants.newImageSize)),  
            # torchvision.transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225]
            # )
        ])
    for idx, annotation in enumerate(data['annotations']):
        imageID = str(annotation['image_id'])
        while len(imageID)<5:
            imageID = "0"+imageID
        image = Image.open(constants.basePath + 
                            "/SODA-D/" + "Images/" + imageID + ".jpg")
        if image.mode == "L":
            continue
        imageIDToImageObj[imageID] = image
        if imageID not in imageIDToBboxes:
            imageIDToBboxes[imageID] = []
        imageIDToBboxes[imageID].append(annotation['bbox'])
        if imageID not in imageIDToTargets:
            imageIDToTargets[imageID] = []
        imageIDToTargets[imageID].append(annotation['category_id'])
    images = []
    targets = []
    for imageID, imageObj in imageIDToImageObj.items():
        images.append(transform(imageObj))
        originalImageWidth, originalImageHeight = Image.open(constants.basePath + 
                            "/SODA-D/" + "Images/" + imageID + ".jpg").size
        scaleFactorWidth = constants.newImageSize/originalImageWidth
        scaleFactorHeight = constants.newImageSize/originalImageHeight
        scaledBboxes = [] 
        for bbox in imageIDToBboxes[imageID]:
            xmin = bbox[0]*scaleFactorWidth
            ymin = bbox[1]*scaleFactorHeight
            xmax = (bbox[0]+bbox[2])*scaleFactorWidth
            ymax = (bbox[1]+bbox[3])*scaleFactorHeight
            scaledBboxes.append([xmin, ymin, xmax, ymax])
        targets.append({
            'boxes': torch.as_tensor(scaledBboxes),
            'labels': torch.as_tensor(imageIDToTargets[imageID])
        })
    file1 = "sodaImages4096Test"
    file2 = "sodaTargets4096Test"
    torch.save(images, constants.basePath + "/SODA-D/"+file1+".pt")
    torch.save(targets, constants.basePath + "/SODA-D/"+file2+".pt")

def transformDHDTrafficDataset():
    imageIDToImageObj = {}
    imageIDToBboxes = {}
    imageIDToTargets = {}
    file = open(constants.basePath + 
        '/dhd_pedestrian/ped_traffic/annotations/'+
        'dhd_pedestrian_traffic_train.json')
    data = json.loads(file.read())
    transform = torchvision.transforms.Compose([ 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(
                size=(constants.newImageSize, constants.newImageSize)),  
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    for idx, annotation in enumerate(data['annotations']):
        imageID = str(annotation['image_id'])
        image = Image.open(constants.basePath + 
            '/dhd_traffic/images/train/'+ imageID + '.jpg')
        if image.mode == "L":
            continue
        imageIDToImageObj[imageID] = image
        if imageID not in imageIDToBboxes:
            imageIDToBboxes[imageID] = []
        imageIDToBboxes[imageID].append(annotation['bbox'])
        if imageID not in imageIDToTargets:
            imageIDToTargets[imageID] = []
        imageIDToTargets[imageID].append(annotation['category_id'])
    images = []
    targets = []
    for imageID, imageObj in imageIDToImageObj.items():
        images.append(transform(imageObj))
        originalImageWidth, originalImageHeight = Image.open(constants.basePath + 
            '/dhd_traffic/images/train/'+ imageID + '.jpg').size
        scaleFactorWidth = constants.newImageSize/originalImageWidth
        scaleFactorHeight = constants.newImageSize/originalImageHeight
        scaledBboxes = [] 
        for bbox in imageIDToBboxes[imageID]:
            xmin = bbox[0]*scaleFactorWidth
            ymin = bbox[1]*scaleFactorHeight
            xmax = (bbox[0]+bbox[2])*scaleFactorWidth
            ymax = (bbox[1]+bbox[3])*scaleFactorHeight
            scaledBboxes.append([xmin, ymin, xmax, ymax])
        targets.append({
            'boxes': torch.as_tensor(scaledBboxes),
            'labels': torch.as_tensor(imageIDToTargets[imageID])
        })
    file1 = "dhdImages4096Train"
    file2 = "dhdTargets4096Train"
    torch.save(images, constants.basePath + "/dhd_traffic/"+file1+".pt")
    torch.save(targets, constants.basePath + "/dhd_traffic/"+file2+".pt")

if __name__=="__main__":
    transformDHDTrafficDataset()
    