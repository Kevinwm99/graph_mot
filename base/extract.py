"""
Phu V April 23th 2021
13:47:!5
This is the code base for extract the the bounding boxes, object centers, object offsets, and the features for each bounding box.

Need to build the dataset first, extract all the detections from the dataset. After that form a graph base on those detections.
Let short-term frame rate= 2 , long-term frame rate = 10.
For each tracklet, find the collection of long-term and short-term representation (bounding box or center).
For each tracklet, we will have two kind of graphs

Basically we only need to write one function for short-term and long-term: Sampler(). Then for each identity we only need to sample 2 times.

Input: collections of frames

Outputs: bb[x1,y1,x2,y2], [c_x,c_y],
"""

# GPU settings
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "8,9"

import argparse
import torch
import json
import time
import cv2
from sklearn import metrics
from scipy import interpolate
import numpy as np
from torchvision.transforms import transforms as T
import matplotlib.pyplot as plt

from models.model import create_model, load_model
from datasets.dataset.jde import DetDataset, collate_fn
from utils.utils import xywh2xyxy, ap_per_class, bbox_iou
from opts import opts
from models.decode import mot_decode
from utils.post_process import ctdet_post_process


def test_det(
        opt,
        batch_size=12,
        img_size=(1088, 608),
        iou_thres=0.5,
        print_interval=40,
):
    data_cfg = opt.data_cfg
    f = open(data_cfg)
    data_cfg_dict = json.load(f)
    f.close()
    nC = 1
    test_path = data_cfg_dict['test']
    dataset_root = data_cfg_dict['root']
    print(opt.gpus[0])
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    #model = torch.nn.DataParallel(model)
    model = model.to(opt.device)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    # model.eval()
    Dataset = get_dataset(opt.dataset, opt.task)
    # Get dataloader
    transforms = T.Compose([T.ToTensor()])
    dataset = DetDataset(dataset_root, test_path, img_size, augment=False, transforms=transforms)
    train_dataset = Dataset(opt, dataset_root, test_path, img_size, augment=False, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=8, drop_last=False,
                                             collate_fn=collate_fn)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=8, drop_last=False)
    mean_mAP, mean_R, mean_P, seen = 0.0, 0.0, 0.0, 0
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    outputs, mAPs, mR, mP, TP, confidence, pred_class, target_class, jdict = \
        [], [], [], [], [], [], [], [], []
    AP_accum, AP_accum_count = np.zeros(nC), np.zeros(nC)
    for batch_i, (imgs, targets, paths, shapes, targets_len) in enumerate(dataloader):
