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
import numpy as np



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

# numpy_img = numpy.asarray(PIL_img)


def arithmetic_or(x,y):
    return x + y - x*y

def normalize_tensor_0_1(x):
    min_x = torch.min(x)
    range_x = torch.max(x) - min_x
    if range_x > 0:
        normalised = (x - min_x) / range_x
    else:
        normalised = 0*x
    return normalised



# show_tensor(torch.sigmoid(alpha*pred + pred.mean()), show_img = True)
#    show_tensor(torch.sigmoid(alpha*target + target.mean()) , show_img = True)
#     pred = pred / pred.sum(0).expand_as(pred)
    # show_tensor(x, show_img = True)
# torch.sigmoid(alpha*pred).requires_grad_(True)
# pred and target are tensors here

def tensor_to_grayscale(img):
    return ( img[:,0,:,:] + img[:,1,:,:] + img[:,2,:,:] )/3    
    
 #    x= torch.sigmoid(alpha*(pred - beta)); show_tensor(x, show_img= True)
'''Same as F1_loss above but it is performed using boolean algebra'''
def F1_loss_torch(pred, target, f1_inv=True, alpha = 1E6 ):   # alpha 1E6 ensures hard thresholding
    
    epsilon = 1e-10 # epsilon used to prevent overflow    
    pred = tensor_to_grayscale(pred) # using, pred[:,0,:,:], only the first channel gave slightly higher result 
    target = tensor_to_grayscale(target)
    
    pred   = torch.sigmoid(1E6*(pred - pred.mean() )).requires_grad_(True)    
    target = torch.sigmoid(1E6*(target - target.mean() )).requires_grad_(False)     
    
    pred = normalize_tensor_0_1(pred)
    target = normalize_tensor_0_1(target)
    # show_tensor(Fp, 'Fp')    
    
    N = arithmetic_or(pred, target)  # logical
    Tp = pred * target
    Fn = torch.abs(target - Tp)
    Fp = torch.abs(pred - Tp)
    Tn = torch.abs(N - arithmetic_or(arithmetic_or(Tp, Fp), Fn) )
    T_n = torch.sum(Tn, dtype=torch.float32) # can be usd to calculate accuracy
    T_p = torch.sum(Tp, dtype=torch.float32)
    F_p = torch.sum(Fp, dtype=torch.float32)
    F_n = torch.sum(Fn, dtype=torch.float32)    
    
    if f1_inv: # to be used in optimization    
        F1 = 1 - 2*(T_p + epsilon)/(2*T_p + F_p + F_n - epsilon) # epsilon = 0 #1e-10  # used to handle extreme values, like, division by zero
       # the minus sign (before epsilon) in the dominator will ensure that F1 is 3 (high value) when there is a zero division, as the objetive is minimization               
    else: 
        F1 = 2*T_p /( 2*T_p + F_n + F_p ) 
    
    return F1

def F1_loss_numpy(pred, target): 
    epsilon = 1e-10
    pred = pred[:,:,0]  # using only the red channel
    target = target[:,:,0]

    N = np.logical_or(pred, target)  # logical 
    Tp = np.logical_and(pred, target) 
    Fn = np.bitwise_xor(target, Tp) # element-wise subtraction in pytorch 
    Fp = np.bitwise_xor(pred, Tp)        
    xx= np.logical_or(np.logical_or(Tp,Fp), Fn)
    Tn = np.bitwise_xor(N, xx)

    precision = Tp.sum()/(Tp.sum()+ Fp.sum() +epsilon)
    recall = Tp.sum()/(Tp.sum()+ Fn.sum()) 
    kk = 2*Tp.sum()+ Fn.sum()+ Fp.sum()
    if kk==0:
        return 0, 0, 0, 0
    else:
        F1 = 2*Tp.sum() / kk
    accuracy = (Tp.sum()+Tn.sum())/N.sum()
 
    return F1, accuracy, precision, recall


def show_tensor(img, fname):
    to_pil = transforms.ToPILImage() 
    img1  = to_pil(img.cpu()) # we can also use test_set[1121][0].numpy()        
    img1.show()        
    img1.save('/home/malrawi/Desktop/GAN_seg_img_414/'+ fname +'.png')        
       

#model = torch.nn.Linear(10, 1)
#x = torch.randn(1, 2)
##y = torch.rand([5, 3, 10, 10])- .5 
##x = torch.rand([5, 3, 10, 10])- .2
#target = torch.randn(1, 2)
#output = model(x)
#loss = F1_loss_prime(output, target)
#loss.backward()
#print(model.weight.grad)




#torch.manual_seed(1)
#y = torch.rand([5, 3, 10, 10])- .5 
#x = torch.rand([5, 3, 10, 10])- .2
#x.requires_grad_(True) 
#y.requires_grad_(True)
## x = Image.open('/home/malrawi/Desktop/My Programs/PyTorch-GANs/data/text_segmentation/test/B/img_614.png')
## y = Image.open('/home/malrawi/Desktop/My Programs/PyTorch-GANs/data/text_segmentation/test/A/img_614.png')
## trsfm = transforms.Compose([
###            transforms.Resize((32,32)),
##            transforms.ToTensor(),                       
##            ])
## x = trsfm(x).unsqueeze(dim=0)
## y = trsfm(y).unsqueeze(dim=0)
#F1_prime  = F1_loss_prime(x, y, reduce=True, F1_inverse=False,
#                          alpha = 1100, beta = 220) # -0.08 threshold 
#
#F1  = F1_loss(x, y, reduce=True, F1_inverse=False, thresh_val=0.2)
#
#
#print(F1_prime)
#print(F1)
#print(F1.is_leaf)
#print(F1_prime.is_leaf)
#
#model = torch.nn.Linear(10, 1)
#pred = model(x)
#target = model(y)
#loss = F1_loss_prime(pred, target)
#loss.backward()
#print(model.weight.grad)



#to_pil = transforms.ToPILImage() 
#to_pil(torch.sigmoid(1000*x-500)).show()
#


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