"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
import cv2

def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32) / 255.0

    img -= np.array(mean, dtype=np.float32)
    img /= np.array(variance, dtype=np.float32)
    return img

def denormalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def resize_aspect_ratio(img, square_size=2560, interpolation=cv2.INTER_CUBIC, mag_ratio=1.0, target_size=None, round=True):
    height, width = img.shape[0:2]

    max_sz = max(height, width)
    # magnify image size
    if target_size is None:
        target_size = mag_ratio * max_sz

    # set original image size
    if target_size > square_size:
        target_size = square_size
    
    ratio = target_size / max_sz

    target_w = int(width * ratio)

    # check rounded width and adjust ratio again
    if round and target_w % 32 != 0:
        # round width
        target_w = target_w + (32 - target_w % 32)
        ratio = target_w / width

    target_h = int(height * ratio)

    resized = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    return resized, ratio

def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img
