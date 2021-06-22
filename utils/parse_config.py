#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 18:51:20 2021

@author: yeyanan
"""

def parse_model_config(path):
    with open(path, 'r') as fp:
        lines = fp.readlines()
    lines = [x for x in lines if x != '\n' and not x.startswith('#')]#not>and>or
    lines = [x.strip() for x in lines]
    module_defs = []
    for line in lines:
        if line.startswith('['):
            module_defs.append({})#append null dic
            module_defs[-1]['type'] = line[1:-1]#dic[] =  str ... -3,-2,-1   [0,2] include 0 not include 2
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split('=')
            module_defs[-1][key.strip()] = value.strip()
    """
    [{'type': 'net', 'batch': '16', 'subdivisions': '1', 'width': '416', 'height': '416', 'channels': '3', 
      'momentum': '0.9', 'decay': '0.0005', 'angle': '0', 'saturation': '1.5', 'exposure': '1.5', 'hue': '.1',
      'learning_rate': '0.001', 'burn_in': '1000', 'max_batches': '500200', 'policy': 'steps', 'steps': '400000,450000', 
      'scales': '.1,.1'}, {'type': 'convolutional', 'batch_normalize': '1', 'filters': '32', 'size': '3', 'stride': '1', 
      'pad': '1', 'activation': 'leaky'}, {'type': 'convolutional', 'batch_normalize': '1', 'filters': '64', 
     'size': '3', 'stride': '2', 'pad': '1', 'activation': 'leaky'},
    {'type': 'convolutional', 'batch_normalize': '1', 'filters': '32', 'size': '1', 'stride': '1', 'pad': '1', 'activation': 'leaky'}, ]
    """
 
    return module_defs
    

def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
#    options['gpus'] = '0,1,2,3'
#    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()#strip delete space in front and back
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')#split = gone return list
        options[key.strip()] = value.strip()
    
    return options
    