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
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('Device:', device)
# Load model
# model = attempt_load('yolov7/weights/yolov7-w6-pose.pt', map_location=device)
# Switch to evaluation mode, map_location=device
# model.eval()

# print('Number of classes:', model.yaml['nc'])
# print('Number of keypoints:', model.yaml['nkpt'])


def show_image(img, figsize): 
    w, h, c = figsize
    k = 2
    # plt.figure(figsize=figsize)
    # plt.axis('off')
    # plt.show()
    name_window = 'test_img'
    cv2.namedWindow(name_window, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name_window, k * h, k * w)
    cv2.imshow(name_window, img)
    cv2.waitKey(0)


# read original image
orig_img = cv2.imread('yolov7/test_img.jpg')
# orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
shape = orig_img.shape
print('Original image', shape)
show_image(orig_img, shape)

cv2.destroyAllWindows()
print("END")
