#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:27:13 2019

@author: malrawi
"""

import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

import PIL.ImageOps 
from numpy import random

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="horse2zebra", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=100, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between saving model checkpoints')
parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')

opt = parser.parse_args()

opt.dataset_name = 'text_segmentation'# 'synthtext'
opt.img_height = 256
opt.img_width = 256
opt.batch_test_size = 5
opt.p_color_augment = 0 # 0.25
opt.AMS_grad = False
opt.use_GT = True
#opt.lr= 0.000002
#opt.b1 = 0.9
# opt.b2 = 
# opt.n_epochs = 200
# opt.decay_epoch =  100
# opt.checkpoint_interval = 10
print(opt)


class NoneTransform(object):
    """ Does nothing to the image, to be used instead of None    
    Args:
        image in, image out, nothing is done
    """
        
    def __call__(self, image):       
        return image

class Invert_B_img(object):
    """ Does nothing to the image, to be used instead of None    
    Args:
        image in, image out, nothing is done
    """
        
    def __call__(self, image):                         
        image = image.point(lambda p: 255-p if p>8 and p<64 else 0 ) # invert         
#        
        
        return image


def random_seeding(seed_value, state, cuda_rng_state):    
    
    np.random.seed(seed_value)
    random.seed(seed_value)    
    torch.random.initial_seed()    
    torch.manual_seed(seed_value)   
    
    if cuda: 
        torch.cuda.manual_seed_all(seed_value)
        torch.cuda.set_rng_state(cuda_rng_state)     
    torch.set_rng_state(state)   

#    cuda_rng_state = torch.cuda.get_rng_state()
#    state = torch.get_rng_state()  
#    seed_value = 12345
#    random_seeding(seed_value, state, cuda_rng_state)        
#    random_seeding(seed_value, state, cuda_rng_state)    
#             
        
    

def get_loaders():
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
                           unaligned=False, 
                           gt=opt.use_GT,
                           p_color_augment= opt.p_color_augment
                           ), 
                    batch_size=opt.batch_size, 
                    shuffle=True,  num_workers=opt.n_cpu)
    
    val_dataloader = DataLoader(ImageDataset("../data/%s" % opt.dataset_name, 
                            transform = transforms_val,                           
                            unaligned=False, mode='test', 
                            gt=opt.use_GT
                            ),
                            batch_size=opt.batch_test_size, 
                            shuffle=True, 
                            num_workers=0,
                            worker_init_fn = torch.initial_seed
                            )
    
    return dataloader, val_dataloader



#           tsfm= transforms.ToPILImage()

def sample_images(imgs, batches_done, use_max=False):
    """Saves a generated sample from the test set"""
     
    real_A_pos = Variable(imgs['A'].type(Tensor))
    fake_B_pos = G_AB(real_A_pos)    
    real_A_neg = Variable(imgs['A_neg'].type(Tensor))
    fake_B_neg = G_AB(real_A_neg)    
    
    img_sample = torch.cat((real_A_pos.data, fake_B_pos.data,
                            real_A_neg.data, fake_B_neg.data), 0)
    save_image(img_sample, 'images/%s/%s.png' % (opt.dataset_name, batches_done), nrow=5, normalize=True)        
    
    if use_max:
        for i in range(len(fake_B_pos)):
            fake_B_neg.data[i] = torch.max(fake_B_pos.data[i], fake_B_neg.data[i])
        img_sample = torch.cat((real_A_pos.data, fake_B_pos.data,
                            real_A_neg.data, fake_B_neg.data), 0)
        save_image(img_sample, 'images/%s/%s_max.png' % (opt.dataset_name, batches_done), nrow=5, normalize=True)
    
    




''''''''''''' Main Program'''''''''''''''''''

cuda = True if torch.cuda.is_available() else False
dataloader, val_dataloader = get_loaders()
# Create sample and checkpoint directories
os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2**4, opt.img_width // 2**4)

# Initialize generator and discriminator
G_AB = GeneratorResNet(res_blocks=opt.n_residual_blocks)
G_BA = GeneratorResNet(res_blocks=opt.n_residual_blocks)
D_A = Discriminator()
D_B = Discriminator()

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load('saved_models/%s/G_AB_%d.pth' % (opt.dataset_name, opt.epoch)))
    G_BA.load_state_dict(torch.load('saved_models/%s/G_BA_%d.pth' % (opt.dataset_name, opt.epoch)))
    D_A.load_state_dict(torch.load('saved_models/%s/D_A_%d.pth' % (opt.dataset_name, opt.epoch)))
    D_B.load_state_dict(torch.load('saved_models/%s/D_B_%d.pth' % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Loss weights
lambda_cyc = 10
lambda_id = 0.5 * lambda_cyc

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()),
                                lr=opt.lr, betas=(opt.b1, opt.b2),
                                amsgrad=opt.AMS_grad) # amsgrad originally was false
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                                 amsgrad=opt.AMS_grad)  # amsgrad originally was false
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                                 amsgrad=opt.AMS_grad)  # amsgrad originally was false

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()


# ----------
#  Training
# ----------
#           tsfm= transforms.ToPILImage()
prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = Variable(batch['A'].type(Tensor))
        real_B = Variable(batch['B'].type(Tensor))
        print('A shape', real_A.shape, 'B shape', real_B.shape, '\n')

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G =    loss_GAN + \
                    lambda_cyc * loss_cycle + \
                    lambda_id * loss_identity

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(dataloader),
                                                        loss_D.item(), loss_G.item(),
                                                        loss_GAN.item(), loss_cycle.item(),
                                                        loss_identity.item(), time_left))

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:            
            sample_images(next(iter(val_dataloader)), batches_done, use_max=True) # another instance


    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), 'saved_models/%s/G_AB_%d.pth' % (opt.dataset_name, epoch))
        torch.save(G_BA.state_dict(), 'saved_models/%s/G_BA_%d.pth' % (opt.dataset_name, epoch))
        torch.save(D_A.state_dict(), 'saved_models/%s/D_A_%d.pth' % (opt.dataset_name, epoch))
        torch.save(D_B.state_dict(), 'saved_models/%s/D_B_%d.pth' % (opt.dataset_name, epoch))


for batch_idx, imgs in enumerate(val_dataloader):
    sample_images(imgs, batch_idx, use_max=True) # another instance
    
    