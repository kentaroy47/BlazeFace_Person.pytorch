#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import stuff
import os
import numpy as np
import torch
import torch.utils.data as data
from itertools import product as product
import time

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
import pandas as pd

# import dataset
from utils.dataset import VOCDataset, DatasetTransform, make_datapath_list, Anno_xml2list, od_collate_fn


# In[3]:


# load files
vocpath = "../VOCdevkit/VOC2007"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(vocpath, "person")

# make Dataset
voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']
color_mean = (104, 117, 123)  # (BGR)の色の平均値
input_size = 300  # 画像のinputサイズを300×300にする

## DatasetTransformを適応
transform = DatasetTransform(input_size, color_mean)
transform_anno = Anno_xml2list(voc_classes)

train_dataset = VOCDataset(train_img_list, train_anno_list, phase = "train", transform=transform, transform_anno = transform_anno)
val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=DatasetTransform(
    input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))

batch_size = 50

train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn, num_workers=8)

val_dataloader = data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, collate_fn=od_collate_fn, num_workers=8)


# In[ ]:




