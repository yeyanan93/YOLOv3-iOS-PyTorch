#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 23:19:02 2021

@author: yeyanan
"""

import torch
import sys

def to_cpu(tensor):
    return tensor.detach().cpu()

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
        
def bbox_wh_iou(wh1, wh2):
    
    wh2 = wh2.t()
    #wh2   torch.Size([2, 19])
    #wh1  torch.Size([2])  tensor([3.6250, 2.8125])
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
       
    #w1   torch.Size([])   w2 torch.Size([19])
    #torch.min(w1, w2)   torch.Size([19]) 就是取出较小值
    #inter_area          torch.Size([19])
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    
    return inter_area / union_area

def bbox_iou(box1, box2, x1y1x2y2=True):
    

    #torch.Size([47, 4])
    #二维的第一个就是框的数量
    #第二个就是xywh
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        #b1_x1     torch.Size([33])
        box1_x1, box1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        box1_y1, box1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        box2_x1, box2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        box2_y1, box2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        
    x1 = torch.max(box1_x1,box2_x1)
    y1 = torch.max(box1_y1,box2_y1)
    
    x2 = torch.min(box1_x2,box2_x2)
    y2 = torch.min(box1_y2,box2_y2)
    
    
    intersection = torch.clamp(x2-x1,min=0) * torch.clamp(y2-y1,min=0)
    
    
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    
    return intersection / (box1_area + box2_area - intersection + 1e-16)

        
def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    
    #pred_boxes    torch.Size([4, 3, 12, 12, 4])
    #pred_cls      torch.Size([4, 3, 12, 12, 80])
    #target        torch.Size([26, 6])
    #anchors       torch.Size([3, 2])
    #ignore_thres  0.5
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor
    
    nB = pred_boxes.shape[0] # batchsieze 4
    nA = pred_boxes.shape[1] # 每个格子对应了多少个anchor
    nG = pred_boxes.shape[2] # gridsize
    nC = pred_cls.shape[-1]  # 类别的数量
    
    # Output tensors
    #obj_mask       torch.Size([4, 3, 15, 15])
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)  # obj，anchor包含物体, 即为1，默认为0 考虑前景
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1) # noobj, anchor不包含物体, 则为1，默认为1 考虑背景
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0) # 类别掩膜，类别预测正确即为1，默认全为0
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0) # 预测框与真实框的iou得分
    tx = FloatTensor(nB, nA, nG, nG).fill_(0) # 真实框相对于网格的位置
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0) 
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)
    
    # Convert to position relative to box
    #torch.Size([20, 4])
    target_boxes = target[:, 2:6] * nG #target中的xywh都是0-1的，可以得到其在当前gridsize上的xywh
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    
    #anchors   torch.Size([3, 2])
    #anchor    torch.Size([2])
    #gwh       torch.Size([46, 2])
    #ious   torch.Size([3, 19])
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors]) #每一种规格的anchor跟每个标签上的框的IOU得分
    #torch.max(a,0)返回每一列中最大值的那个元素，且返回索引（返回最大元素在这一列的行索引）
    #torch.max      return    tuple  (max, max_indices)
    
    #best_ious  torch.Size([25])
    best_ious, best_n = torch.max(ious,0)
    
    
    #target的0是第几张照片，1是物体的种类
    #.long() is equivalent to .to(torch.int64)
    #b tensor([0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3])
    #target_labels   tensor([ 0, 30, 24, 52, 52, 48, 39, 60, 46, 48, 20, 20])
    b, target_labels = target[:, :2].long().t()
    #gx tensor([7.7034, 1.8598, 5.0641, 5.9130, 7.6285, 8.6823, 5.0084])
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    #i  tensor([7, 1, 5, 5, 7, 8, 5])
    gi, gj = gxy.long().t() #位置信息，向下取整了
    obj_mask[b, best_n, gj, gi] = 1 # 实际包含物体的设置成1
    noobj_mask[b, best_n, gj, gi] = 0 # 相反
    
    # Set noobj mask to zero where iou exceeds ignore threshold
    #ious   torch.Size([19, 3])
    #anchor_ious  tensor([0.8293, 0.2803, 0.0712]) 
    for i, anchor_ious in enumerate(ious.t()): # IOU超过了指定的阈值就相当于有物体了
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0
       
    #floor()    Returns a new tensor with the floor of the elements of input
    #the largest integer less than or equal to each element.
    tx[b, best_n, gj, gi] = gx - gx.floor() # 根据真实框所在位置，得到其相当于网络的位置
    ty[b, best_n, gj, gi] = gy - gy.floor()
    
    #Width and height
    #anchors[best_n][:, 0]是anchors的w
    #这里的log就是ln
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    #One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1 #将真实框的标签转换为one-hot编码形式
    # Compute label correctness and iou at best anchor 计算预测的和真实一样的索引
    
    #pred_cls      torch.Size([4, 3, 13, 13, 80])
    #argmax()
    #Returns the indices of the maximum value of all elements in the input tensor
    #argmax(-1)即可以理解位在最后一维上找到最大值的索引
    #pred_cls[b, best_n, gj, gi]      torch.Size([21, 80])
    #pred_cls[b, best_n, gj, gi].argmax(-1)找21个最大值    torch.Size([21])
    
    #。float（）就是把int转换成float
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    #与真实框相匹配的预测框之间的iou值
    #x1y1x2y2是两个点还是xywh
    #iou_scores[b, best_n, gj, gi].shape     torch.Size([38])
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)
    
    
    #有物体就是1.  没物体就是0.
    tconf = obj_mask.float()
    
    
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
























