#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 23:19:02 2021

@author: yeyanan
"""

import torch

def load_classes(path):
    with open(path, 'r') as fp:
        names = fp.read().split('\n')[0:-1]
        #read() read the whole file in one time return a str  split return a list
    return names

def weights_init_normal(m):
    
    classname = m.__class__.__name__#获取类名 返回str
    if classname.find("Conv") != -1:
        #find()方法检测字符串中是否包含子字符串，如果包含子字符串返回开始的索引值，否则返回-1
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)#torch.nn.init.normal_(tensor, mean=0, std=1)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)