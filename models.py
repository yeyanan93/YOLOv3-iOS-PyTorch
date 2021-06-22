#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 22:11:39 2021

@author: yeyanan
"""
import torch.nn as nn
from utils.parse_config import *
import sys

def create_modules(module_defs):
    
    hyperparams = module_defs.pop(0)#该方法返回从列表中移除的元素对象        module_defs为pop后剩下的   
    output_filters = [int(hyperparams["channels"])]#[3]
    """
    [{'type': 'convolutional', 'batch_normalize': '1', 'filters': '32', 'size': '3', 'stride': '1', 'pad': '1', 'activation': 'leaky'}, 
    {'type': 'convolutional', 'batch_normalize': '1', 'filters': '64', 'size': '3', 'stride': '2', 'pad': '1', 'activation': 'leaky'}, ]
    
    """
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        
        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            stride=int(module_def["stride"])
            pad = (kernel_size - 1) // 2   #" / "  表示浮点数除法，返回浮点结果;  " // " 表示整数除法,返回不大于结果的一个最大的整数
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,#filter的个数
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,#例如当 padding =1 的时候，如果原来大小为 3 × 3 ，那么之后的大小为 5 × 5
                    bias=not bn#是否将一个 学习到的 bias 增加输出中，默认是 True 卷积之后，如果接BN操作，最好是不设置偏置，因为不起作用
                )
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
                #torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                #num_features为输入batch中图像的channle数
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
                
        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)
            
        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())
            
        elif module_def["type"] == "route": # 输入1：26*26*256 输入2：26*26*128  输出：26*26*（256+128）
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())
            
        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            #range(start, stop, step)       start: 计数从 start 开始       stop: 计数到 stop 结束，但不包括 stop。
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
#            modules.add_module(f"yolo_{module_i}", yolo_layer)
        
        # Register module list and number of output filters
        module_list.append(modules)#这个把每个layer加入一次
        output_filters.append(filters)#这个把每个layer加入一次
        
    #hyperparams是超参数
    #module_list是一个ModuleList()    
        
    return hyperparams, module_list


class Upsample(nn.Module):

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
    
class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()
        
class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.img_dim = img_dim
    

class Darknet(nn.Module):
    def __init__(self, config_path):
        super(Darknet, self).__init__()
        #module_defs is a list       element is dic which is the parameter of each layer
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        
    def forward(self, x, targets=None):
        #torch.Size([4, 3, 384, 384])
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        #modules_defs已经pop net
        """
        i     0
        module_def    {'type': 'convolutional', 'batch_normalize': '1', 'filters': '32', 'size': '3', 'stride': '1', 'pad': '1', 'activation': 'leaky'}
        module      Sequential(
                (conv_0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (batch_norm_0): BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True)
                (leaky_0): LeakyReLU(negative_slope=0.1)
                )
        """
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                #conv_layer = module[0]
                #print(conv_layer.weight.data.shape) tensor   torch.Size([32, 3, 3, 3])         
                x = module(x)
                
                sys.exit()


        
        
        
        
        return x, targets
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    