#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:08:08 2019

@author: malrawi
"""

import torchvision.transforms as transforms
from datasets import ImageDataset
from torch.utils.data import DataLoader
import torch
from models import GeneratorResNet
from F1_loss import F1_loss_numpy
import numpy as np
from PIL import ImageChops 
from matplotlib import pyplot as plt

def test_GAN_AB(folder_model, model_name, val_dataloader):  
    n_residual_blocks = 9 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G_AB = GeneratorResNet(res_blocks=n_residual_blocks)
    cuda = True if torch.cuda.is_available() else False
    if cuda: G_AB = G_AB.cuda()
    G_AB.load_state_dict(torch.load(folder_model + model_name ) )
    f1 = []
    with torch.no_grad():
        for i, imgs_batch in enumerate(val_dataloader): 
            real_A = imgs_batch['A'].to(device) 
            real_A_neg = imgs_batch['A_neg'].to(device) 
            real_B = imgs_batch['B'].to(device)            
            B_gt = show_tensor(real_B)
            if double_gan:
                B_gan = show_tensor( G_AB( G_AB(real_A) ))
                B_gan_neg = show_tensor(G_AB( G_AB(real_A_neg) ))
            else:
                B_gan = show_tensor( G_AB(real_A) )
                B_gan_neg = show_tensor(G_AB(real_A_neg) )
            B_gan = ImageChops.add_modulo(B_gan, B_gan_neg)            
            
            x1= F1_loss_numpy(np.asarray(B_gan)>10, 
                                np.asarray(B_gt)>5, # needed to threshold the GT as well 
                                # np.asarray(B_gt)>5
                                )[0] 
            f1.append(x1)
        f1_av = np.mean(f1)
    
    return f1_av, f1       
        
def show_tensor(img, show=False):
    to_pil = transforms.ToPILImage() 
    img1  = to_pil(img.cpu().squeeze()) # we can also use test_set[1121][0].numpy()    
    if show: img1.show()
    return img1


dataset_name = 'text_segmentation256' 
batch_test_size = 1 
transforms_val =[ transforms.ToTensor(), 
                  transforms.Normalize((0.5,0.5,0.5), (.25,.25,.25)) 
                 ]
val_dataloader = DataLoader(ImageDataset("../data/%s" % dataset_name, 
                            transform = transforms_val, aligned=True, 
                            mode='test', data_mode = '' ),
                            batch_size= batch_test_size, shuffle=False, num_workers=1                            
                            )

double_gan = True
folder_model = './saved_models/unaligned-text_segmentation256-may26/'
model_name = 'G_AB_300.pth'
f1_av, f1 = test_GAN_AB(folder_model, model_name, val_dataloader)
print(f1_av)

# x= [ i for i in range(len(f1))]
# plt.scatter(x, f1)


