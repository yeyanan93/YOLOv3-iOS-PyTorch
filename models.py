#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 22:11:39 2021

@author: yeyanan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.parse_config import *
from utils.utils import build_targets, to_cpu
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
            #layers   [-1, 61]
            layers = [int(x) for x in module_def["layers"].split(",")]                            
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())
            
        elif module_def["type"] == "yolo":
            #[6, 7, 8]
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            #[10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            #[(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            #range(start, stop, step)       start: 计数从 start 开始       stop: 计数到 stop 结束，但不包括 stop
            #[(116, 90), (156, 198), (373, 326)]
            anchors = [anchors[i] for i in anchor_idxs]
            #80            
            num_classes = int(module_def["classes"])
            #416
            img_size = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        
        # Register module list and number of output filters
        module_list.append(modules)#这个把每个layer加入一次
        #convolutional shortcut route 有自己的filters 而upsample yolo用的上一层的filters 即channels的数量
        output_filters.append(filters)#这个把每个layer加入一次
        
    #hyperparams是超参数
    #module_list是一个ModuleList()
        
    return hyperparams, module_list

class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)#3
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.grid_size = 0 #
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1#confidence 正样本loss的调节权重
        self.noobj_scale = 100#confidence 负样本loss的调节权重（此值越大，模型的抗误检能力越强
        self.metrics = {}
        
    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        """
        print(g)  16
        print(torch.arange(g))  tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])
        print(torch.arange(g).repeat(g, 1))
        tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
        ......
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]])
    
        print(torch.arange(g).repeat(g, 1).view([1, 1, g, g]))
        tensor([[[[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
          ......
          [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]]]])
        
        """
        
        self.grid_x = torch.arange(g).repeat(g, 1).view(1, 1, g, g).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view(1, 1, g, g).type(FloatTensor)
        """
        FloatTensor把[(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]转换成下面
        
        tensor([[ 3.6250,  2.8125],
                [ 4.8750,  6.1875],
                [11.6562, 10.1875]])  
        """
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        """
        self.scaled_anchors[:, 0:1]
        tensor([[ 3.6250],
                [ 4.8750],
                [11.6562]])
    
        self.anchor_w
        tensor([[[[ 3.6250]],

                 [[ 4.8750]],

                 [[11.6562]]]])
        """
        self.anchor_w = self.scaled_anchors[:, 0:1].view(1, self.num_anchors, 1, 1)
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))
        
    def forward(self, x, targets=None, img_dim=None):
        
        #类型转换, 将list,numpy转化为tensor
        #torch.Tensor默认是torch.FloatTensor是32位浮点类型数据，torch.LongTensor是64位整型
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor
        
        
        
        self.img_dim = img_dim
        
        #Tensor.size() 返回 torch.Size() 对象， Tensor.shape 等价于 Tensor.size()
        #torch.Size([4, 255, 16, 16])
        #num_samples=4 grid_size=16
        num_samples = x.shape[0]
        grid_size = x.shape[2]
        
        #prediction torch.FloatTensor
        #prediction torch.Size([4, 3, 11, 11, 85])    
        #pytorch中view函数的作用为重构张量的维度
        #3*85   =  255
        #如果想要变得连续使用contiguous方法，如果Tensor不是连续的，则会重新开辟一块内存空间保证数据是在内存中是连续的，
        #如果Tensor是连续的，则contiguous无操作。
        prediction = x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()
        
        #x y w h pred_conf       torch.Size([4, 3, 14, 14])
        #pred_cls                torch.Size([4, 3, 14, 14, 80])
        
        x = torch.sigmoid(prediction[:,:,:,:,0]) # Center x
        y = torch.sigmoid(prediction[:,:,:,:,1])  # Center y
        w = prediction[:,:,:,:,2]
        h = prediction[:,:,:,:,3]
        pred_conf = torch.sigmoid(prediction[:,:,:,:,4])  # Conf
        pred_cls = torch.sigmoid(prediction[:,:,:,:,5:])  # Cls pred.
        
        #grid_size就是yolo层输出的size
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)
            
            
        # Add offset and scale with anchors #特征图中的实际位置
        #prediction[:,:,:,:,:4].shape  torch.Size([4, 3, 12, 12, 4])
        pred_boxes = FloatTensor(prediction[:,:,:,:,:4].shape)
        
        #x.data  torch.Size([4, 3, 16, 16])
        #x.data  y.data   w.data   h.data靠网络自己学习
        pred_boxes[:,:,:,:,0] = x.data + self.grid_x
        pred_boxes[:,:,:,:,1] = y.data + self.grid_y
        pred_boxes[:,:,:,:,2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[:,:,:,:,3] = torch.exp(h.data) * self.anchor_h
        
        
        
        output = torch.cat( 
            (
                #torch.Size([4, 363, 4])
                pred_boxes.view(num_samples, -1, 4) * self.stride, #还原到原始图中
                #torch.Size([4, 363, 1])
                pred_conf.view(num_samples, -1, 1),
                #torch.Size([4, 363, 80])
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )
        #torch.Size([4, 363, 85])
        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )
            
            #mse_loss  将逐个元素求差,然后求平方,再求和,再求均值
            #obj_mask         torch.Size([4, 3, 14, 14])
            #x[obj_mask].shape   torch.Size([27])
            #不加这两行就会出现如下错误
            #UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.
            #bool instead.
            obj_mask = obj_mask.bool()
            noobj_mask = noobj_mask.bool()
            
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask]) # 只计算有目标的
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            
            #bce_loss  Creates a criterion that measures the Binary Cross Entropy between the target and the output
            #loss_conf_obj    tensor(0.8041, grad_fn=<BinaryCrossEntropyBackward>)
            #loss_conf_noobj  tensor(0.7290, grad_fn=<BinaryCrossEntropyBackward>)
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])

            
            #tensor(0.8041, grad_fn=<BinaryCrossEntropyBackward>)
            #self.obj_scale = 1   self.noobj_scale = 100
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            
            #14个位置有框        torch.Size([14, 80])
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask]) #分类损失
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls #总损失
            
            # Metrics
            #class_mask[obj_mask]       torch.Size([26])  
            cls_acc = 100 * class_mask[obj_mask].mean()#预测类别的准确率
            conf_obj = pred_conf[obj_mask].mean()
            
            print(class_mask[obj_mask].shape)
            sys.exit()
            
            return output, total_loss
        
        
        
        
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
        

    

class Darknet(nn.Module):
    def __init__(self, config_path):
        super(Darknet, self).__init__()
        #module_defs is a list       element is dic which is the parameter of each layer
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        #hasattr() 函数用于判断对象是否包含对应的属性。有就返回true  没有就false
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]

        
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
            elif module_def["type"] == "route":
                #torch.cat是将两个张量（tensor）拼接在一起
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                
                layer_i = int(module_def["from"])

                """
                torch.Size([4, 64, 208, 208])
                torch.Size([4, 64, 208, 208])
                torch.Size([4, 64, 208, 208])
                """
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                #把img_dim换掉
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                """
                torch.Size([4, 432, 85])
                torch.Size([4, 1728, 85])
                torch.Size([4, 6912, 85])
                """
                yolo_outputs.append(x)
            layer_outputs.append(x)
        #yolo_outputs   torch.Size([4, 12348, 85])
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
            
        return yolo_outputs if targets is None else (loss, yolo_outputs)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    