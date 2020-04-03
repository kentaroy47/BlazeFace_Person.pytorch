#!/usr/bin/env python
# coding: utf-8

# # import

# In[1]:


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
#import pandas as pd
from math import sqrt as sqrt
from itertools import product as product


# # backbone
# from blazeface-pytorch

class BlazeBlock(nn.Module):
    def __init__(self, inp, oup1, oup2=None, stride=1, kernel_size=5):
        super(BlazeBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        
        # double-block is used when oup2 is specified.
        self.use_double_block = oup2 is not None
        # pooling is used when stride is not 1
        self.use_pooling = self.stride != 1
        # change padding settings to insure pixel size is kept.
        if self.use_double_block:
            self.channel_pad = oup2 - inp
        else:
            self.channel_pad = oup1 - inp
        padding = (kernel_size - 1) // 2
        
        # mobile-net like convolution function is defined.
        self.conv1 = nn.Sequential(
            # dw
            # https://discuss.pytorch.org/t/depthwise-and-separable-convolutions-in-pytorch/7315
            # if groups=inp, it acts as depth wise convolution in pytorch
            nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=padding, groups=inp, bias=True),
            nn.BatchNorm2d(inp),
            # piecewise-linear convolution.
            nn.Conv2d(inp, oup1, 1, 1, 0, bias=True),
            nn.BatchNorm2d(oup1),
        )
        self.act = nn.ReLU(inplace=True)
        
        # for latter layers, use resnet-like double convolution.
        if self.use_double_block:
            self.conv2 = nn.Sequential(
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup1, oup1, kernel_size=kernel_size, stride=1, padding=padding, groups=oup1, bias=True),
                nn.BatchNorm2d(oup1),
                # pw-linear
                nn.Conv2d(oup1, oup2, 1, 1, 0, bias=True),
                nn.BatchNorm2d(oup2),
            )

        if self.use_pooling:
            self.mp = nn.MaxPool2d(kernel_size=self.stride, stride=self.stride)

    def forward(self, x):
        h = self.conv1(x)
        if self.use_double_block:
            h = self.conv2(h)

        # skip connection
        if self.use_pooling:
            x = self.mp(x)
        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), 'constant', 0)
        return self.act(h + x)

# initialize weights.
def initialize(module):
    # original implementation is unknown
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight.data, 1)
        nn.init.constant_(module.bias.data, 0)
    
class BlazeFace(nn.Module):
    """Constructs a BlazeFace model
    the original paper
    https://sites.google.com/view/perception-cv4arvr/blazeface
    """

    def __init__(self, channels=24):
        super(BlazeFace, self).__init__()
        # input..128x128
        self.features = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, stride=2, padding=1, bias=True), # pix=64
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            BlazeBlock(channels, channels, channels),
            BlazeBlock(channels, channels, channels),
            BlazeBlock(channels, channels*2, channels*2, stride=2), # pix=32
            BlazeBlock(channels*2, channels*2, channels*2),
            BlazeBlock(channels*2, channels*2, channels*2),
            BlazeBlock(channels*2, channels, channels*4, stride=2), # pix=16
            BlazeBlock(channels*4, channels, channels*4),
            BlazeBlock(channels*4, channels, channels*4)
        )
        self.apply(initialize)
    def forward(self, x):
        h = self.features(x)
        return h

    
# for test
#net = BlazeFace()

# # add extra blocks for detection and localization
# originally from ssd.pytorch
class BlazeFaceExtra(nn.Module):
    """Constructs a BlazeFace model
    the original paper
    https://sites.google.com/view/perception-cv4arvr/blazeface
    """
    def __init__(self, channels=24):
        super(BlazeFaceExtra, self).__init__()
            # input..128x128
        self.features = nn.Sequential(
                BlazeBlock(channels*4, channels, channels*4, stride=2), # pix=8
                BlazeBlock(channels*4, channels, channels*4),
                BlazeBlock(channels*4, channels, channels*4)
        )
        self.apply(initialize)
    def forward(self, x):
        h = self.features(x)
        return h
    
class BlazeFaceExtra2(nn.Module):
    """
    for blazeface 256.
    """
    def __init__(self, channels=24):
        super(BlazeFaceExtra2, self).__init__()
            # input..128x128
        self.features = nn.Sequential(
                BlazeBlock(channels*4, channels*2, channels*8, stride=2), # pix=8
                BlazeBlock(channels*8, channels*2, channels*8),
                BlazeBlock(channels*8, channels*2, channels*8)
        )
        self.apply(initialize)
    def forward(self, x):
        h = self.features(x)
        return h

#extras = BlazeFaceExtra()

def make_loc_conf(num_classes=2, bbox_aspect_num=[6, 6], channels=24):
    loc_layers = []
    conf_layers = []
    
    # added more layers.
    loc_layers += [nn.Sequential(nn.Conv2d(channels*4, bbox_aspect_num[0]*4, kernel_size=3, padding=1))]
    conf_layers += [nn.Sequential(nn.Conv2d(channels*4, bbox_aspect_num[0]*num_classes, kernel_size=3, padding=1))]
    
    # 
    loc_layers += [nn.Sequential(nn.Conv2d(channels*4, bbox_aspect_num[1]*4, kernel_size=3, padding=1))]
    conf_layers += [nn.Sequential(nn.Conv2d(channels*4, bbox_aspect_num[1]*num_classes, kernel_size=3, padding=1))]
    
    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)

def make_loc_conf256(num_classes=2, bbox_aspect_num=[6, 6], channels=24):
    loc_layers = []
    conf_layers = []
    
    # added more layers.
    loc_layers += [nn.Sequential(nn.Conv2d(channels*4, bbox_aspect_num[0]*4, kernel_size=3, padding=1))]
    conf_layers += [nn.Sequential(nn.Conv2d(channels*4, bbox_aspect_num[0]*num_classes, kernel_size=3, padding=1))]
    
    # 
    loc_layers += [nn.Sequential(nn.Conv2d(channels*8, bbox_aspect_num[1]*4, kernel_size=3, padding=1))]
    conf_layers += [nn.Sequential(nn.Conv2d(channels*8, bbox_aspect_num[1]*num_classes, kernel_size=3, padding=1))]
                
    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)



# class for generating binding boxes
class DBox(object):
    def __init__(self, cfg):
        super(DBox, self).__init__()
        
        self.image_size = cfg["input_size"]
        # 各sourceの特徴量マップのサイズ
        self.feature_maps = cfg["feature_maps"]
        self.num_priors = len(cfg["feature_maps"]) # number of sources
        self.steps = cfg["steps"] #各boxのピクセルサイズ
        self.min_sizes = cfg["min_sizes"] # 小さい正方形のサイズ
        self.max_sizes = cfg["max_sizes"] # 大きい正方形のサイズ
        self.aspect_ratios = cfg["aspect_ratios"]
        
    def make_dbox_list(self):
        mean = []
        # feature maps = 38, 19, 10, 5, 3, 1
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                # fxf画素の組み合わせを生成
                
                f_k = self.image_size / self.steps[k]
                # 300 / steps: 8, 16, 32, 64, 100, 300
                
                # center cordinates normalized 0~1
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                
                # small bbox [cx, cy, w, h]
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]
                
                # larger bbox
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]
                
                # その他のアスペクト比のdefbox
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
                    
        # convert the list to tensor
        output = torch.Tensor(mean).view(-1, 4)
        
        # はみ出すのを防ぐため、大きさを最小0, 最大1にする
        output.clamp_(max=1, min=0)
        
        return output

def nms(boxes, scores, overlap=0.45, top_k=200):
    """
    overlap以上のディテクションに関して信頼度が高い方をキープする。
    キープしないものは消去。
    
    物体のクラス毎にnmsは実効する。
    ------------------
    inputs:
        scores: bboxの信頼度
        bbox: bboxの座標情報
    
    ------------------
    出力:
        keep:
    """
    
    # returnを定義
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()
    #print(keep.size())
    # keep: 確信度thresholdを超えたbboxの数
    
    # 各bboxの面積を計算
    x1 = boxes[:, 0]
    x2 = boxes[:, 2]
    y1 = boxes[:, 1]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    
    # copy boxes
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()
    
    # sort scores 高い信頼度のものを上に。
    v, idx = scores.sort(0)
    
    # topk個の箱のみ取り出す
    idx = idx[-top_k:]
    
    # indexの要素数が0でない限りループする。
    while idx.numel() > 0:
        i = idx[-1] # 一番高い信頼度のboxを指定
        
        # keep の最後にconf最大のindexを格納
        keep[count] = i
        count += 1
        
        # 最後の一つになったらbreak
        if idx.size(0) == 1:
            break
        # indexをへらす
        idx = idx[:-1]
        
        # ------------------------------------
        # このboxとiouの大きいboxを消していく。
        # ------------------------------------
        
        # torch.index_select(input, dim, index, out=None) → Tensor
        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)
        
        # target boxの最小、最大にclamp
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, min=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, min=y2[i])
        
        # wとhのテンソルサイズをindex一つ減らしたものにする
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)
        
        # clampした状態の高さ、幅を求める
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1
        
        # 幅や高さが負になっているものは0に
        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)
        
        # clamp時の面積を導出
        inter = tmp_w * tmp_h # オーバラップしている面積
        
        # IoU の計算
        # intersect=overlap
        # IoU = intersect部分 / area(a) + area(b) - intersect
        rem_areas = torch.index_select(area, 0, idx) # bbox元の面積
        union = rem_areas + area[i] - inter
        IoU = inter / union

        # IoUがしきい値より大きいものは削除
        idx = idx[IoU.le(overlap)] # leはless than or eqal to
    
    return keep, count


class Detect(Function):
    """
    for inference.
    """
    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def forward(self, loc_data, conf_data, dbox_list):
        """
        SSDの推論結果を受け取り、bboxのデコードとnms処理を行う。
        """
        # 
        num_batch = loc_data.size(0)
        num_dbox = loc_data.size(1)
        num_classes = conf_data.size(2)
        
        # confをsoftmaxを使って正規化
        conf_data = self.softmax(conf_data)
        
        # 出力の方を作成する
        # [batch, class, topk, 5]
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)
        
        # conf_dataを[batch, 8732, classes]から[batch, classes, 8732]に変更
        conf_preds = conf_data.transpose(2, 1)
        
        # batch毎にループ
        for i in range(num_batch):
            # 1. LocとDBoxからBBox情報に変換
            #print("loc", loc_data.shape)
            #print("box", dbox_list.shape)
            decoded_boxes = decode(loc_data[i], dbox_list.to(self.device))
            
            # confのコピー
            conf_scores = conf_preds[i].clone()
            
            # classごとにデコードとNMSを回す。
            for cl in range(1, num_classes): # 背景は飛ばす。
                # 2. 敷地を超えた結果を取り出す
                c_mask = conf_scores[cl].gt(self.conf_thresh) # gt=greater than
                # index maskを作成した。
                # threshを超えると1, 超えなかったら0に。
                # c_mask = [8732]
                
                scores = conf_scores[cl][c_mask]
                
                if scores.nelement() == 0:
                    continue
                    # 箱がなかったら終わり。
                    
                # cmaskをboxに適応できるようにサイズ変更
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # l_mask.size = [8732, 4]
                boxes = decoded_boxes[l_mask].view(-1, 4) # reshape to [boxnum, 4]
                
                # 3. NMSを適応する
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                
                # torch.cat(tensors, dim=0, out=None) → Tensor
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 
                                                 1)
                
        return output # torch.size([batch, 21, 200, 5])

def decode(loc, dbox_list):
    """
    Decode boxes.
    
    DBox(cx,cy,w,h)から回帰情報のΔを使い、
    BBox(xmin,ymin,xmax,ymax)方式に変換する。
    
    loc: [8732, 4] [Δcx, Δcy, Δw, Δheight]
    SSDのオフセットΔ情報
    
    dbox_list: (cx,cy,w,h)
    """
    
    boxes = torch.cat((
    dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, :2],
    dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), dim=1)
    
    # convert boxes to (xmin,ymin,xmax,ymax)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    
    return boxes


class SSD(nn.Module):
    """
    module for ssd-like blazeface.
    phase: train
    - for training.
    phase: inference
    - for inference and evaluation.
    """
    def __init__(self, phase, cfg, channels=24):
        super(SSD, self).__init__()
        
        self.phase = phase
        self.num_classes = cfg["num_classes"]
        
        # call SSD network
        self.blaze = BlazeFace(channels)
        self.extra = BlazeFaceExtra(channels)
        # self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(self.num_classes, cfg["bbox_aspect_num"], channels)
        
        # make Dbox
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()
        
        # use Detect if inference
        if phase == "inference":
            self.detect = Detect()
            
    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()
        
        # compute blazeface block
        x = self.blaze(x)
        sources.append(x)
        # compute extra block
        x = self.extra(x)
        sources.append(x)
        
        # compute loc and cof
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # Permuteは要素の順番を入れ替え
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
        # convの出力は[batch, 4*anker, fh, fw]なので整形しなければならない。
        # まず[batch, fh, fw, anker]に整形
        
        # locとconfの形を変形
        # locのサイズは、torch.Size([batch_num, 34928])
        # confのサイズはtorch.Size([batch_num, 183372])になる
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        # さらにlocとconfの形を整える
        # locのサイズは、torch.Size([batch_num, 8732, 4])
        # confのサイズは、torch.Size([batch_num, 8732, 21])
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)
        # これで後段の処理につっこめるかたちになる。
        
        output = (loc, conf, self.dbox_list)
        
        if self.phase == "inference":
            # Detectのforward
            return self.detect(output[0], output[1], output[2])
        else:
            return output

class SSD256(nn.Module):
    """
    module for ssd-like blazeface with 256 pix input.
    larger than the original inplementation.
    """
    def __init__(self, phase, cfg, channels=24):
        super(SSD256, self).__init__()
        
        self.phase = phase
        self.num_classes = cfg["num_classes"]
        
        # call SSD network
        self.blaze = BlazeFace(channels=channels)
        self.extra = BlazeFaceExtra(channels=channels)
        self.extra2 = BlazeFaceExtra2(channels=channels)
        # self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf256(self.num_classes, cfg["bbox_aspect_num"], channels=channels)
        
        # make Dbox
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()
        
        # use Detect if inference
        if phase == "inference":
            self.detect = Detect()
            
    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()
        
        # compute blazeface block
        x = self.blaze(x)
        # compute extra block
        x = self.extra(x)
        sources.append(x)
        x = self.extra2(x)
        sources.append(x)
        
        # compute loc and cof
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # Permuteは要素の順番を入れ替え
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
        # convの出力は[batch, 4*anker, fh, fw]なので整形しなければならない。
        # まず[batch, fh, fw, anker]に整形
        
        # locとconfの形を変形
        # locのサイズは、torch.Size([batch_num, 34928])
        # confのサイズはtorch.Size([batch_num, 183372])になる
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        # さらにlocとconfの形を整える
        # locのサイズは、torch.Size([batch_num, 8732, 4])
        # confのサイズは、torch.Size([batch_num, 8732, 21])
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)
        # これで後段の処理につっこめるかたちになる。
        
        output = (loc, conf, self.dbox_list)
        
        if self.phase == "inference":
            # Detectのforward
            return self.detect(output[0], output[1], output[2])
        else:
            return output



