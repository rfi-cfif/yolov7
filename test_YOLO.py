# import requests
import cv2
import torch
from torchvision import transforms
# import numpy as np
# import matplotlib.pyplot as plt

from models.experimental import attempt_load 
from utils.datasets import letterbox 
from utils.general import non_max_suppression_kpt, xywh2xyxy
from utils.plots import output_to_keypoint, plot_skeleton_kpts, plot_one_box

# WEIGHTS_URL = 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt'
# open("weights/yolov7-w6-pose.pt", "wb").write(requests.get(WEIGHTS_URL).content)


# set gpu device if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('Device:', device)
# Load model
model = attempt_load('yolov7/weights/yolov7-w6-pose.pt', map_location=device)
# Switch to evaluation mode, map_location=device
model.eval()

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
# show_image(orig_img, shape)

img = letterbox(orig_img, 640, stride=64, auto=True)[0]
print('Resized image', img.shape)
# show_image(img, img.shape)

print('Anchors:', model.yaml['anchors'])

# transform to tensor
img_ = transforms.ToTensor()(img)
# add dimension
img_ = torch.unsqueeze(img_, 0)
print('Transformed to tensor image:', img_.shape)
# send the picture to the calculating device
img_ = img_.to(device).float()

with torch.no_grad():
    pred, _ = model(img_)
print('Predictions shape:', pred.shape)

pred = non_max_suppression_kpt(pred, 
                               conf_thres=0.25, 
                               iou_thres=0.65, 
                               nc=model.yaml['nc'], 
                               nkpt=model.yaml['nkpt'], 
                               kpt_label=True)
print('Detected poses:', len(pred))
print('Prediction shape:', pred[0].shape)
# print('pred', pred)
def plot_pose_prediction(img : cv2.Mat, pred : list, thickness=2, 
                         show_bbox : bool=True) -> cv2.Mat:
    bbox = xywh2xyxy(pred[:,2:6])
    for idx in range(pred.shape[0]):
        plot_skeleton_kpts(img, pred[idx, 7:].T, 3)
        if show_bbox:
            plot_one_box(bbox[idx], img, label='person', line_thickness=thickness)
    return bbox

pred = output_to_keypoint(pred)
# print()
# print('pred2', pred)
res = plot_pose_prediction(img, pred)
cmx1, cmy1, cmx2, cmy2 = res[0]
print('res', res)
print(cmx1, cmy1, cmx2, cmy2, cmx1.dtype)
cv2.circle(img, (int(cmx1 + (cmx2 - cmx1) / 2), int(cmy1 + (cmy2 - cmy1) // 2)), 5, (255,255,255), -1)
show_image(img, img.shape)

cv2.destroyAllWindows()
print("END")
