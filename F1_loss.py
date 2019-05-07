#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:46:58 2019

@author: malrawi
Mohammed Al-Rawi


https://discuss.pytorch.org/t/implementing-element-wise-logical-and-tensor-operation/33596/2
https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/
"""
import torch

'''Calculates the F1 loss based on two inputs, output and the target
Input args- 
    - output: batch of images as tensor
    - target: batch of images as tensor
    The input is a batch of images with the following structure BxCxWxH, where B
    is the batch size, C is the number of channels (which is usually 3 for RGB images), W & H
    are the image width and height, respectively.
    - reduce=['Channel', 'Batch'] .. averaging the scores over channels and/or batches
    if no reduce is intended, the input should be an empty list
    
Output args- 
- F1 score... here, it is a BxC array, the F1 score for each image in the batch, and
each band as well. If one needs the F1 loss over each image, then, F1.mean(dim=1), see reduce . 
- accuracy

'''
def F1_loss(output, target, reduce=['Channel']):   
    
    thresh_val = 0
    output = torch.gt(output, thresh_val).byte()
    target = torch.gt(target, thresh_val).byte()
    N = output | target  # logical
    Tp = output & target
    Fn = target - Tp # element-wise subtraction in pytorch 
    Fp = output - Tp     
    Tn = N - (Tp | Fp | Fn) 
    Tp = torch.sum(Tp, dim=(2,3), keepdim=True, dtype=torch.float32).squeeze() # summing over x,y, keeping the batch and the channel dim
    Tn = torch.sum(Tn, dim=(2,3), keepdim=True, dtype=torch.float32).squeeze()
    Fp = torch.sum(Fp, dim=(2,3), keepdim=True, dtype=torch.float32).squeeze()
    Fn = torch.sum(Fn, dim=(2,3), keepdim=True, dtype=torch.float32).squeeze()
    P = Tp/(Tp+Fp) #  torch.div(Tp, Tp+ Fp)  # Precision
    R =    Tp/(Tp+Fn) # torch.div(Tp, Tp + Fn) # Recall
    F1 = 2*P*R/(P+R)
    if reduce[0]=='Channel':
        F1 = F1.mean(dim=1)
    if reduce[1] == 'Batch':
        F1 = F1.mean()
        
    accuracy = (Tp + Tn)/torch.sum(N, dim=(2,3), keepdim=True, dtype=torch.float32).squeeze()

    return F1, accuracy 


y = torch.rand([5, 3, 10, 10])-.5 
x = torch.rand([5, 3, 10, 10])- .2

F1, acc = F1_loss(x, y, reduce=[])

print(F1)
