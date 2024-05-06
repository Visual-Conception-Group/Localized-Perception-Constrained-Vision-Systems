from PIL import Image
import torch

k = 2 # number of patches sampled at a time (for patchGD only)
batchSize = 4
patchSize = 512
pandaImageSize = 1024
aidImageSize = 512
innerIterations = 2 # for patchGD only
# numClasses = 6 # for PANDA classification
numClasses = 91 # for coco detection
# numClasses = 11 # for SODA-D classification
epochs = 100
accumulationSteps = 2
extractedFeatureLength = 2048
Image.MAX_IMAGE_PIXELS = 9933130000
basePath = "/DATA/niyati17032/RA"
lr = 0.0001
# device = "cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark =  True
torch.backends.cudnn.enabled =  True

fcosNumberOfHeads = 5
H = [64, 32, 16, 8, 4]
W = [64, 32, 16, 8, 4]
cocoImageSize = 512
newImageSize = 4096
numberOfImagesToMerge = 36
numberOfImagesInOneRow = 6

m = newImageSize//patchSize
n = newImageSize//patchSize
