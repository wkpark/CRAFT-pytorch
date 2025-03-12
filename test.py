"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict
def copyStateDict(state_dict):
    if "craft" in state_dict.keys():
        state_dict = state_dict["craft"]
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.2, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.3, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--gpu', default=False, action='store_true', help='Use GPU for inference')
parser.add_argument('--canvas_size', default=3200, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.3, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
parser.add_argument('images', nargs='*', help='Image files')

args = parser.parse_args()


""" For test images in a folder """
result_folder = './result/'

if len(args.images) > 0:
    images = args.images

    image_list = []
    for image in images:
        if os.path.exists(image):
            image_list.append(image)
        else:
            path = os.path.join(args.test_folder, image)
            if os.path.exists(path):
                image_list.append(path)
            else:
                raise Exception("File not found:", image)
else:
    image_list, _, _ = file_utils.get_files(args.test_folder)

if not os.path.isdir(result_folder):
    os.mkdir(result_folder)


def test_net(canvas_size, mag_ratio, net, image, text_threshold, link_threshold, low_text, poly, device, estimate_num_chars=False,
        refine_net=None, verbose=False, options={}):

    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR,
        mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    img_resized = img_resized.astype(np.float32)

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    x = x.to(device)

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    del x

    # make score and link map
    score_text = y[0, ..., 0].detach().cpu().numpy()
    score_link = y[0, ..., 1].detach().cpu().numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, ...,0].detach().cpu().numpy()

    del y, feature

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys, mapper = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if verbose : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text



if __name__ == '__main__':
    # load net
    t = time.time()

    if args.gpu:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
            print('Neither CUDA nor MPS are available - defaulting to CPU. Note: This module is much faster with a GPU.')
    else:
        device = 'cpu'
        print('use CPU.')

    state_dict = copyStateDict(torch.load(args.trained_model, map_location=device, weights_only=True))

    net = CRAFT()     # initialize

    net.load_state_dict(state_dict)

    print('Loading weights from checkpoint (' + args.trained_model + ')')

    net = net.to(device)
    if device != 'cpu':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location=device, weights_only=False)))
        if device != 'cpu':
            refine_net = torch.nn.DataParallel(refine_net)

        refine_net.eval()
        args.poly = True

    t1 = time.time()
    print("loading time : {}s".format(t1 - t))
    t = t1

    options = vars(args)
    options["result_folder"] = result_folder

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)
        filename, file_ext = os.path.splitext(os.path.basename(image_path))

        options["filename"] = filename

        bboxes, polys, score_text = test_net(
            args.canvas_size, args.mag_ratio, net, image, args.text_threshold, args.link_threshold, args.low_text, args.poly, device,
            refine_net=refine_net, verbose=args.show_time, options=options,
        )

        # save score text
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))
