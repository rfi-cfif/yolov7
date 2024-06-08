import requests
import cv2
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from models.experimental import attempt_load 
from utils.datasets import letterbox 
from utils.general import non_max_suppression_kpt, xywh2xyxy
from utils.plots import output_to_keypoint, plot_skeleton_kpts, plot_one_box

# WEIGHTS_URL = 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt'
# open("weights/yolov7-w6-pose.pt", "wb").write(requests.get(WEIGHTS_URL).content)


# set gpu device if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)
# Load model
model = attempt_load('weights/yolov7-w6-pose.pt', map_location=device) 
# Switch to evaluation mode, map_location=device
model.eval()

print('Number of classes:', model.yaml['nc'])
print('Number of keypoints:', model.yaml['nkpt'])


print("END")
