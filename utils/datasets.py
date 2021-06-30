#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:39:46 2021

@author: yeyanan
"""


import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    
    pad1 = dim_diff // 2 #a//b得到的值为a/b得到的最小整数
    pad2 = dim_diff - dim_diff // 2
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    img = F.pad(img, pad, "constant", value=pad_value)
    #torch.nn.functional.pad(input, pad, mode='constant', value=0)
    #(left,right,top,bottom)

    return img, pad#img tensor [3, 640, 640]  pad tuple

def resize(image, size):
    #squeeze()是降维  unsqueeze()是升维
    #F.interpolate是上下采样
    #通过这个函数就可以得到想要的图片大小
    
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)

    return image

class ListDataset(Dataset):
    
    def __init__(self,list_path,img_size=416,multiscale=True):
        
        # Read img path
        with open(list_path, 'r') as file:# list_path is trainvalno5k.txt
            self.img_files = file.readlines()# readlines will return a list
            
        # Convert label path from img path    
        self.label_files = []
        for path in self.img_files:
            path = path.replace('images', 'labels').replace(".jpg", ".txt")
            self.label_files.append(path)
            
        self.img_size = img_size
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0         #为了multiscale,每多少batch改变imagesize
        
        
    def __getitem__(self,index):#用来获取一些索引的数据，返回数据集中第i个样本。
        
        
        # ---------
        #  Image
        # ---------
        
        img_path = self.img_files[index].rstrip()#rstrip delete space in back
        img_path = '/Users/yeyanan/YOLO/2_代码/YOLOv3-iOS-PyTorch/data/coco' + img_path
        
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))#img是一个tensor格式[[[R]],[[G]],[[B]]]
        c, h, w = img.shape#img.shape     a subclass of tuple   [3, 480, 640]
        
        img, pad = pad_to_square(img,0)
        #img tensor 正方形
        #pad是一个tuple  (0, 0, 80, 80)
        c_padded, h_padded, w_padded = img.shape
        
        # ---------
        #  Label
        # ---------
        label_path = self.label_files[index].rstrip()
        label_path = '/Users/yeyanan/YOLO/2_代码/YOLOv3-iOS-PyTorch/data/coco' + label_path
        boxes = np.loadtxt(label_path).reshape(-1, 5)
        #loadtxt return numpy.ndarray  reshape return(m,n)m行n列   -1代表不固定，随着另外一个维度改变
        #loadtxt dtype float64
        """
        loadtxt 返回ndarray 长这样
        [[ 1.  2.  3.  4.]
         [ 2.  3.  4.  5.]
         [ 3.  4.  5.  6.]
         [ 4.  5.  6.  7.]]
        """
        boxes = torch.from_numpy(boxes)
        
        x_box = w * boxes[:,1]
        y_box = h * boxes[:,2]
        w_box = w * boxes[:,3]
        h_box = h * boxes[:,4]
        #x,y是框的中点坐标
        #w是框的宽，h是框的高
        
        
        boxes[:,1] = (pad[1] + x_box) / w_padded
        boxes[:,2] = (pad[2] + y_box) / h_padded
        boxes[:,3] = w_box / w_padded
        boxes[:,4] = h_box / h_padded
        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = boxes
        #由于pad，需要转换下框的位置
        
#        print(img.shape)
#        print(img.shape)
#        img = img.numpy()
#        img = np.swapaxes(img,0,1)
#        img = np.swapaxes(img,1,2)
#        print(img.shape)
#        
#        plt.figure()
#        fig, ax = plt.subplots(1)
#        rect = patches.Rectangle((w_padded * boxes[0,1]-w_padded * boxes[0,3]/2, h_padded * boxes[0,2]-h_padded * boxes[0,4]/2), w_padded * boxes[0,3], h_padded * boxes[0,4], linewidth=1, edgecolor='r', facecolor='none')
#        ax.add_patch(rect)
#        ax.imshow(img)
        #画框      检验框的位置正确与否
        #imshow的输入只能为(n, m) or (n, m, 3) 即h w 3
#        print(img_path, img, targets)
        
        
        return img_path, img, targets
        #targets就是框
        """
         /Users/yeyanan/YOLO/2_代码/YOLOv3-iOS-PyTorch/data/coco/images/train2014/COCO_train2014_000000000025.jpg
        """
        """
        tensor([[[0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         ...,
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.]],

        [[0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         ...,
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.]],

        [[0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         ...,
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.]]])
        """
        
        """
        tensor([[23.0000,  0.7703,  0.4931,  0.3359,  0.4643],
        [23.0000,  0.1860,  0.7673,  0.2063,  0.0862]], dtype=torch.float64)
        """
        
    def collate_fn(self, batch):
        #batch就是（img_path, img, targets）tuple组成的batch_size     list
        #zip(*batch)就是把img_path放在一起成一个tuple,把img放在一起成一个tuple,把targets放在一起成一个tuple
        
#        print(batch[0])
#        print(batch[1])
#        print(batch[2])
        paths, imgs, targets = list(zip(*batch))
        #zip(*batch) return zip
        #list(zip(*batch)) return list   
        """
        >>> letters = ['b', 'a', 'd', 'c']
        >>> numbers = [2, 4, 3, 1]
        >>> data1 = list(zip(letters, numbers))
        >>> data1
        [('b', 2), ('a', 4), ('d', 3), ('c', 1)]
        """
        
        """
        paths,imgs,targets都是tuple
        """
        # Remove empty placeholder targets
        #既去除了空的targets，又把tuple变为list
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        #torch.cat((x, x), 0)是按行来拼接targets里面的tensor
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            #choice() 方法返回一个列表，元组或字符串的随机项。
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        #resize是上面定义的函数
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1        

        #paths tuple
        #imgs tensor
        #targets tensor
        return paths, imgs, targets
        
    def __len__(self):
        return len(self.img_files)#返回所有照片的数量
        
    
    
    
    
    
    
    
    
    
        