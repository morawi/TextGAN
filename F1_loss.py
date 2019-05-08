#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:46:58 2019

@author: malrawi
Mohammed Al-Rawi


https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/

"""
import torch
from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models

'''Calculates the F1 loss based on two inputs, pred and target
Input args- 
    - pred: batch of images as tensor
    - target: batch of images as tensor
    The input is a batch of images with the following structure BxCxWxH, where B
    is the batch size, C is the number of channels (which is usually 3 for RGB images), W & H
    are the image width and height, respectively.
    - reduce= True or False .. averaging the scores over channels if True (default)
    else, if no reduce is intended, False should be used
    - alpha is a scale and shift used for the sigmoid default is alpha = 1100, beta = 220, used
    to threshold the image. The search for the best alpha and beta values can be done
    by running F1_loss with some threshold value, and then, trying F1_loss_prime with 
    a few alpha and beta values until the two functions give the same F1_inverse 
    value (or F1 value). Our experiments showed that the relation between beta and 
    threshold_val (of Fa_loss) can have the form beta=10*threshold_val 
    
Output args- 
- F1 score... here, it is a BxC array, the F1 score for each image in the batch, and
each band as well. If one needs the F1 loss over each image, then, F1.mean(dim=1), see reduce . 
- accuracy

Example
torch.manual_seed(1)
y = torch.rand([5, 3, 256, 256])- .5 
x = torch.rand([5, 3, 256, 256])- .2
x.requires_grad_(True)
y.requires_grad_(True)
F1, acc = F1_loss(x, y, reduce=True, F1_inverse=True)
F1_prime  = F1_loss_prime(x, y, reduce=True, F1_inverse=True, alpha = 1100, beta = 220) 

print(F1)
print(F1_prime)


The difference betwen F1_loss and F1_loss_prime is that the later depends on
boolean algebra, that is, to perform logical operations via arithmetic 
operations which makes it suitable for backprobagation. A threshold value of 
-0.34 in F1_loss_prime is equivalent to binarizing via pred = torch.gt(pred, thresh_val).byte()
in F1_loss. The threshold value in F1_loss_prime, alpha, should be selected carefully. That is, 
for F1_loss_prime, the value of alpha affects the thresholding.

'''
def F1_loss(pred, target, reduce= True, thresh_val = 0, F1_inverse=True):   
    
    pred = torch.gt(pred, thresh_val).byte()
    target = torch.gt(target, thresh_val).byte()
    N = pred | target  # logical
    Tp = pred & target
    Fn = target - Tp # element-wise subtraction in pytorch 
    Fp = pred - Tp     
    Tn = N - (Tp | Fp | Fn) 
    Tp = torch.sum(Tp, dim=(2,3), keepdim=True, dtype=torch.float32).squeeze() # summing over x,y, keeping the batch and the channel dim
    Tn = torch.sum(Tn, dim=(2,3), keepdim=True, dtype=torch.float32).squeeze()
    Fp = torch.sum(Fp, dim=(2,3), keepdim=True, dtype=torch.float32).squeeze()
    Fn = torch.sum(Fn, dim=(2,3), keepdim=True, dtype=torch.float32).squeeze()    
    F1_inv= 0.5* (2*Tp+Fp+Fn)/Tp
    
    if F1_inverse==False:
        F1 = 1/ F1_inv
    else:
        F1 = F1_inv
    
    if reduce == True:        
        F1 = F1.mean()
    else:
        F1 = F1.mean(dim=0) #returns measure for each band
    
    accuracy = (Tp + Tn)/torch.sum(N, dim=(2,3), keepdim=True, dtype=torch.float32).squeeze()

    return F1, accuracy 


'''Same as F1_loss above but it is performed using boolean algebra'''
def F1_loss_prime(pred, target, reduce= True, alpha = 1100, beta = 220, F1_inverse=True):   
    # epsilon = 0 #1e-10  # used to handle extreme values, like, division by zero
    pred=torch.sigmoid(alpha*pred-beta)
    target = torch.sigmoid(alpha*target-beta)    
    N = arithmetic_or(pred, target)  # logical
    Tp = pred * target
    Fn = target - Tp 
    Fp = pred - Tp     
    Tn = N - arithmetic_or(arithmetic_or(Tp, Fp), Fn) 
    Tp = torch.sum(Tp, dim=(2,3), keepdim=True, dtype=torch.float32).squeeze() # summing over x,y, keeping the batch and the channel dim
    Tn = torch.sum(Tn, dim=(2,3), keepdim=True, dtype=torch.float32).squeeze()
    Fp = torch.sum(Fp, dim=(2,3), keepdim=True, dtype=torch.float32).squeeze()
    Fn = torch.sum(Fn, dim=(2,3), keepdim=True, dtype=torch.float32).squeeze()    
    
    F1_inv = 0.5* (2*Tp+Fp+Fn)/(Tp)
    
    if F1_inverse==False:
        F1 = 1/ F1_inv
    else:
        F1 = F1_inv
    
    if reduce == True:        
        F1 = F1.mean()
    else:
        F1 = F1.mean(dim=0) # returns measure for each band
        
    return F1




def arithmetic_or(x,y):
    return x + y - x*y



torch.manual_seed(1)
y = torch.rand([5, 3, 256, 256])- .5 
x = torch.rand([5, 3, 256, 256])- .2

x.requires_grad_(True)
y.requires_grad_(True)
# x = Image.open('/home/malrawi/Desktop/My Programs/PyTorch-GANs/data/text_segmentation/test/B/img_614.png')
# y = Image.open('/home/malrawi/Desktop/My Programs/PyTorch-GANs/data/text_segmentation/test/A/img_614.png')
# trsfm = transforms.Compose([
##            transforms.Resize((32,32)),
#            transforms.ToTensor(),                       
#            ])
# x = trsfm(x).unsqueeze(dim=0)
# y = trsfm(y).unsqueeze(dim=0)
F1_prime  = F1_loss_prime(x, y, reduce=True, F1_inverse=True,
                          alpha = 1100, beta = 220) # -0.08 threshold 



F1, acc = F1_loss(x, y, reduce=True, F1_inverse=True,
                  thresh_val=.2)


print(F1_prime)

print(F1)



#to_pil = transforms.ToPILImage() 
#to_pil(torch.sigmoid(1000*x-500)).show()
#


#P = Tp/(Tp+Fp) #  torch.div(Tp, Tp+ Fp)  # Precision
#R =    Tp/(Tp+Fn) # torch.div(Tp, Tp + Fn) # Recall
#F1 = 2*P*R/(P+R)  # we need the F1 inverse as the loss is a minimization problem



#
#model = models.resnet18(pretrained=True)
#model.fc = torch.nn.Linear(2048, 1 ) # num
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#
#
#loss = 0
#model.train()    
#for epoch in range(1, 5):
#    optimizer.zero_grad()
#    output = model(x)
#    target = model(y)    
#    loss += F1_loss_prime(output,target).item()                      
#    loss.backward()
#    optimizer.step()
#    print(loss)
#        
#
#print(model)
# print(model.parameters())[0].grad)
# print(model.parameters())[0].weight)

#for p in model.parameters():
#    if p.grad is not None:
#        print(p.grad.data)