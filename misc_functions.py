#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:53:36 2019

@author: malrawi
"""
import torchvision.transforms as transforms
from datasets import ImageDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch
import numpy as np
from numpy import random




def binarize_tensor(img):
    thresh_val = img.mean()    
    img = (img>thresh_val).float()*1       
    img = 2*img-1
    
    return img

# choosing the best output between the positive and negative
def reason_images(fake_B_pos, fake_B_neg):
    for i in range(len(fake_B_pos)):
            s_pos = fake_B_pos.data[i].sum()
            s_neg = fake_B_neg.data[i].sum()
            if s_pos>s_neg:
                fake_B_neg.data[i] = fake_B_pos.data[i] 
    return fake_B_neg



def test_performance(Tensor, val_dataloader, G_AB, 
                                  criterion_testing):
    ''' Calculates the overall identitiy loss of the test set '''
    loss_id_B = 0
    with torch.no_grad():
        for batch_idx, imgs in enumerate(val_dataloader):            
            real_B_pos = imgs['B'].type(Tensor)       
            real_A_pos = imgs['A'].type(Tensor)            
            GAN_B_pos = G_AB(real_A_pos)                         
            real_A_neg = imgs['A_neg'].type(Tensor)
            GAN_B_neg = G_AB(real_A_neg)             
            
            fake_B_neg = reason_images(GAN_B_pos, GAN_B_neg)                
            
            loss_id_B += criterion_testing(GAN_B_pos, real_B_pos)
            
       
    print('Identity L1 evaluation all testing samples', loss_id_B.item()/len(val_dataloader.dataset))
    


def sample_images(imgs, batches_done, G_AB, Tensor, opt, use_max=False):
    """Saves a generated sample from the test set"""
    second_pass_gan = True 
    real_A_pos = imgs['A'].type(Tensor)
    fake_B_pos = G_AB(real_A_pos)    
    real_A_neg = imgs['A_neg'].type(Tensor)
    fake_B_neg = G_AB(real_A_neg)    
    
    img_sample = torch.cat((real_A_pos.data, fake_B_pos.data,
                            real_A_neg.data, fake_B_neg.data, 
                            binarize_tensor(fake_B_pos+fake_B_neg) ), 0)
    save_image(img_sample, 'images/%s/%s.png' % (opt.dataset_name, batches_done), 
               nrow=5, normalize=True)        
        
    if second_pass_gan:        
        fake_BB_pos = G_AB(fake_B_pos)            
        fake_BB_neg = G_AB(fake_B_neg)
        img_sample = torch.cat((real_A_pos.data, fake_BB_pos.data,
                        real_A_neg.data, fake_BB_neg.data ), 0)
#        img_sample = torch.cat((fake_BB_pos, fake_BB_neg,
#                        (fake_BB_pos.data+fake_BB_neg.data)/2 ), 0)
        save_image(img_sample, 'images/%s/%s_2nd_gan.png' % (opt.dataset_name, batches_done), 
                   nrow=6, normalize=True)
        



def random_seeding(seed_value, state, cuda_rng_state, cuda):    
    
    np.random.seed(seed_value)
    random.seed(seed_value)    
    torch.random.initial_seed()    
    torch.manual_seed(seed_value)   
#    torch.backends.cudnn.enabled=False
#    torch.backends.cudnn.deterministic=True
    
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
                           data_mode = opt.data_mode,                           
                           p_RGB2BGR_augment= opt.p_RGB2BGR_augment, 
                           p_invert_augment=opt.p_invert_augment, 
                           
                           ), 
                    batch_size=opt.batch_size, 
                    shuffle=True,  
                    num_workers=opt.n_cpu)
    
    val_dataloader = DataLoader(ImageDataset("../data/%s" % opt.dataset_name, 
                            transform = transforms_val,                           
                            aligned=True, # should always be aligned
                            mode='test', 
                            data_mode = opt.data_mode,
                            ),
                            batch_size=opt.batch_test_size, 
                            shuffle=True, 
                            num_workers=1                            
                            )
    
    return dataloader, val_dataloader

#           tsfm= transforms.ToPILImage()
    



