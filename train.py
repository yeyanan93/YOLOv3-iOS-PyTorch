#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:03:10 2021

@author: yeyanan
"""
from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test_data import evaluate
from terminaltables import AsciiTable

import os
import time
import torch
import argparse

from torch.utils.data import DataLoader
from torch.autograd import Variable
import sys

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    args = parser.parse_args()
    
    #device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    # create output and checkpoints directories
    # exist_ok = false may give a error if already exists
    
    os.makedirs('output', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Get data configuration
    data_config = parse_data_config(args.data_config)
    train_path = data_config['train']#train label path is also obtained from image
    valid_path = data_config['valid']#valid label path is also obtained from image
    class_names = load_classes(data_config["names"])
    
    # ############
    # Create model
    # ############
    model = Darknet(args.model_def).to(device)
    model.apply(weights_init_normal)
    #module  apply 会对里面的都进行函数处理  包括自己
    
    # Get dataloader
    dataset = ListDataset(train_path)
    #ListDataset返回的数据   主要是getitem这个函数返回
    
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    #Dataloader就是一个迭代器，最基本的使用就是传入一个 Dataset 对象，它就会根据参数 batch_size 的值生成一个 batch 的数据
    
    optimizer = torch.optim.Adam(model.parameters())#model.parameters()为该实例中可优化的参数
    
    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]
    """
    
    epoch = 1 forward and backward pass of ALL training samples
    
    batch_size = number of training samples in one forward and backward pass
    
    number of iterations = number of passes, each pass using [batch_size] number of samples
    
    e.g. 100 samples, batch_size=20 --> 100/20 = 5 iterations for 1 epoch
    
    """
    
    
    for epoch in range(args.epochs):
        model.train()
        #在使用pytorch构建神经网络的时候，训练过程中会在程序上方添加一句model.train()，作用是启用batch normalization和drop out
        #drop out在前向传播的时候，让某个神经元以一定的概率p停止工作，这样可以使模型泛化性更强，因为它不会太依赖某些局部的特征
        start_time = time.time()#返回当前时间的时间戳
        for batch_i, (paths, imgs, targets) in enumerate(dataloader):
            #paths is tuple
            #dataloader的长度就是有多少batch
            batches_done = len(dataloader) * epoch + batch_i
            
            #imgs        torch.Size([4, 3, 448, 448])
            #targets     torch.Size([26, 5])
            #tensor不能反向传播，variable可以反向传播  nn.module的输入为Variable  varible and tensor 都是torch.FloatTensor
            imgs = torch.autograd.Variable(imgs.to(device))
            targets = torch.autograd.Variable(targets.to(device), requires_grad=False)
            print(imgs.shape)
            print('targets',targets.shape)
            
            #loss   tensor(231.5121, grad_fn=<AddBackward0>)
            #outputs   torch.Size([4, 16128, 85]
            loss, outputs = model(imgs, targets)
            loss.backward()
            print("loss")
            
            if batches_done % args.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()
                print("step")
                
            # ----------------
            #   Log progress
            # ----------------
            
            #---- [Epoch 0/1, Batch 0/15] ----
            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, args.epochs, batch_i, len(dataloader))
            
            #metric_table    [['Metrics', 'YOLO Layer 0', 'YOLO Layer 1', 'YOLO Layer 2']]
            #列表前面加星号作用是将列表解开成多个独立的参数，传入函数。
            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]
            
            # Log metrics at each YOLO layer
            
            for i, metric in enumerate(metrics):
                """
                {'grid_size': '%.6f', 'loss': '%.6f', 'x': '%.6f', 'y': '%.6f', 
                'w': '%.6f', 'h': '%.6f', 'conf': '%.6f', 'cls': '%.6f', 
                'cls_acc': '%.6f', 'recall50': '%.6f', 'recall75': '%.6f', 
                'precision': '%.6f', 'conf_obj': '%.6f', 'conf_noobj': '%.6f'}
                """
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                
                #['13', '26', '52']
                #formats[metric] 是输出的格式   yolo.metrics.get(metric, 0)是具体的数值
                #The get() method returns the value of the item with the specified key.
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                #[['Metrics', 'YOLO Layer 0', 'YOLO Layer 1', 'YOLO Layer 2'], ['grid_size', '16', '32', '64']]
                metric_table += [[metric, *row_metrics]]
            print("evaluation_interval")
        if epoch % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            precision, recall, AP, f1, ap_class = evaluate(
                    model,
                    path=valid_path,
                    iou_thres=0.5,
                    conf_thres=0.5,
                    nms_thres=0.5,
                    img_size=args.img_size,
                    batch_size=8,
            )
            print("evaluate")
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            print("ap_table")
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
        if epoch % args.checkpoint_interval == 0:
            
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
                
            
                
                
            
            
                        
            
            
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    