from ultralytics import YOLO
import constants
import warnings

warnings.filterwarnings('ignore')

def yoloDetection():
    # Load a model
    model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model
    # Run batched inference on a list of images
    _ = model(constants.basePath + "/dummy.jpg", save=True)

if __name__=="__main__":
    yoloDetection()
