#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 23:19:02 2021

@author: yeyanan
"""

import torch
import numpy as np
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
        
def xywh2xyxy(x):
    #tensor.new  创建一个新的Tensor，该Tensor的type和device都和原有Tensor一致，且无内容
    y = x.new(x.shape)    
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    #此函数就是把xywh转化成xyxy
    return y
def get_batch_statistics(outputs, targets, iou_threshold):
    
    batch_metrics = []
    ##print(len(output))   8
    for sample_i in range(len(outputs)):
        
        if outputs[sample_i] is None:
            continue
        
        #output    torch.Size([3921, 7])
        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]
                
        #true_positives     numpy.ndarray
        #len(true_positives)    4823
        true_positives = np.zeros(pred_boxes.shape[0])
        
        
        #把第sample_i照片的targets选出来
        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        
        #target_labels 
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]
            
            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                
                if len(detected_boxes) == len(annotations):
                    break
                
                if pred_label not in target_labels:
                    continue
                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
        
        print(len(true_positives))
        sys.exit()
    
    
    
    return batch_metrics

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    #torch.Size([8, 6300, 85])
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    
    #len tensor 返回Tensor第一位的维度
    #[None, None, None, None, None, None, None, None]
    
    #
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        
        #image_pred  torch.Size([7623, 85])
        #image_pred只是一张图片的预测，所以从三维变成二维
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        #torch.Size([2033, 85])        
        
        if not image_pred.size(0):
            continue
        
        
        
        #image_pred[:, 5:]    torch.Size([7517, 80])
        #torch.max(a,1)返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
        """
        image_pred[:, 5:].max(1)
        torch.return_types.max(
        values=tensor([0.6407, 0.6414, 0.6344,  ..., 0.5431, 0.5430, 0.5395]),
        indices=tensor([53, 53, 64,  ..., 43, 57, 12]))
        """
        #image_pred[:, 5:].max(1)[0]  tensor([0.6207, 0.6044, 0.6045,  ..., 0.5438, 0.5512, 0.5494])
        
        # Object confidence times class confidence
        #score   torch.Size([8791])
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        
        #argsort()   Returns the indices that sort a tensor along a given dimension in ascending order by value
        #默认是升序
        #torch.Size([5277, 85])
        #此语句是对image_pred进行排序处理，把最大值放到最前面
        image_pred = image_pred[(-score).argsort()]
        
        #keepdim whether the output tensor has dim retained or not. Default: False.
        """
        torch.return_types.max(
                values=tensor([[0.6757],
                               [0.6727],
                               [0.6227],
                               ...,
                               [0.5392],
                               [0.5382],
                               [0.5392]]),
        indices=tensor([[20],
                        [20],
                        [26],
                        ...,
                        [33],
                        [76],
                        [33]]))
        """
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        #image_pred[:, :5]     torch.Size([6423, 5])
        #class_confs.float()   torch.Size([6423, 1])
        #class_preds.float()   torch.Size([6423, 1])
        #detections            torch.Size([6423, 7])
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.shape[0]:
            #detections[0, :4]    torch.Size([4])
            #detections[0, :4].unsqueeze(0)    torch.Size([1, 4])
            #bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4])  torch.Size([5382])
            #large_overlap      tensor([False, False, False,  ..., False, False, False])
            
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            
            #label_match  tensor([ True,  True,  True,  ..., False, False, False])
            label_match = detections[0, -1] == detections[:, -1]
            
            #&     The & symbol is a bitwise AND operator.  两个都是true 就是 true
            invalid = large_overlap & label_match
            #weights    torch.Size([0, 1])
            weights = detections[invalid, 4:5]
            
            # torch.sum(input, dim, out=None) dim：求和的方向。若input为2维tensor矩阵，dim=0，对列求和；dim=1，对行求和
            # Merge overlapping bboxes by order of confidence
            #对所有预测差不多的框，根据列的第五  做平均处理
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            #把之前用过的框都去掉
            detections = detections[~invalid]
            
        #keep_boxes    [tensor([-40.1647, -26.1947,  74.3477,  59.3861,   0.5597,   0.6293,  66.0000]),
        if keep_boxes:
            #image_i   就是第几张照片
            #torch.stack(keep_boxes).shape    torch.Size([310, 7])
            output[image_i] = torch.stack(keep_boxes)
            
    #print(len(output))   8
    #print(output[0].shape)   torch.Size([3897, 7])
    return output
       
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
        box1_x1, box1_y1, box1_x2, box1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        box2_x1, box2_y1, box2_x2, box2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        
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
























