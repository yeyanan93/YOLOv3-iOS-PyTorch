#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 21:39:18 2021

@author: yeyanan
"""

import tqdm
from utils.utils import *
from utils.datasets import *
from torch.utils.data import DataLoader
from torch.autograd import Variable


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    #model.eval()，pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值
    model.eval()
    
    dataset = ListDataset(path)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    
    #Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息
    for batch_i, (paths, imgs, targets) in enumerate(dataloader):
        
        
        #labels是一个一维list，里面装的事物体的种类
        labels += targets[:, 1].tolist()
        
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        #由相对值转换成绝对值
        targets[:, 2:] *= img_size

        #imgs   torch.Size([8, 3, 512, 512])
        imgs = torch.autograd.Variable(imgs.type(Tensor), requires_grad=False)
        
        with torch.no_grad():
            
            #outputs      torch.Size([8, 6300, 85])
            outputs = model(imgs)
            #conf_thres=0.5     nms_thres=0.5
            #print(len(output))   8
            #print(output[0].shape)   torch.Size([3897, 7])
            print("non_max_suppression")
#            if(batch_i==1){
#                    print(outputs)
#                    sys.exit()
#                    }
        
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        """
        [[array([0., 0., 0., ..., 0., 0., 0.]), tensor([0.5653, 0.5636, 0.5613,  ..., 0.5034, 0.5048, 0.5003]), 
        """
        print("get_batch_statistics")
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        print(sample_metrics)
        
        
    #print(len(sample_metrics))  image 的个数
    
    #print(true_positives)  [0. 0. 0. ... 0. 0. 0.]
    
    print(sample_metrics)
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    
    return precision, recall, AP, f1, ap_class









