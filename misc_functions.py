#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:53:36 2019

@author: malrawi
"""
import torchvision.transforms as transforms
from datasets import *
from torch.utils.data import DataLoader
# from torchvision import datasets
import torch
import numpy as np
from numpy import random



def random_seeding(seed_value, state, cuda_rng_state, cuda):    
    
    np.random.seed(seed_value)
    random.seed(seed_value)    
    torch.random.initial_seed()    
    torch.manual_seed(seed_value)   
    
    if cuda: 
        torch.cuda.manual_seed_all(seed_value)
        torch.cuda.set_rng_state(cuda_rng_state)     
    torch.set_rng_state(state)   
    

def get_loaders(opt):
    # Image transformations
    transforms_gan = [    
    #        transforms.Resize(int(opt.img_height*1.12), Image.BICUBIC),
    #        transforms.RandomCrop((opt.img_height, opt.img_width)),
    #        transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
    ]
    
    transforms_val = [ 
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    
    # Training data loader
    dataloader = DataLoader(ImageDataset("../data/%s" % opt.dataset_name, 
                           transform=transforms_gan,                            
                           aligned=opt.aligned, 
                           gt=opt.use_white_GT,                           
                           p_RGB2BGR_augment= opt.p_RGB2BGR_augment, 
                           p_invert_augment=opt.p_invert_augment
                           ), 
                    batch_size=opt.batch_size, 
                    shuffle=True,  
                    num_workers=opt.n_cpu)
    
    val_dataloader = DataLoader(ImageDataset("../data/%s" % opt.dataset_name, 
                            transform = transforms_val,                           
                            aligned=True, # should always be aligned
                            mode='test', 
                            gt=opt.use_white_GT
                            ),
                            batch_size=opt.batch_test_size, 
                            shuffle=True, 
                            num_workers=1                            
                            )
    
    return dataloader, val_dataloader

#           tsfm= transforms.ToPILImage()
    

# choosing the best output between the positive and negative
def reason_images(fake_B_pos, fake_B_neg):
    for i in range(len(fake_B_pos)):
            s_pos = fake_B_pos.data[i].sum()
            s_neg = fake_B_neg.data[i].sum()
            if s_pos>s_neg:
                fake_B_neg.data[i] = fake_B_pos.data[i] 
    return fake_B_neg


