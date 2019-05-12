#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:27:13 2019

@author: malrawi

https://hardikbansal.github.io/CycleGANBlog/
https://github.com/eriklindernoren/PyTorch-GAN

"""

import argparse
import os
import numpy as np
import itertools
import datetime
import time
from torch.autograd import Variable
from models import *
from utils import *
import torch
from F1_loss import F1_loss_prime
from misc_functions import random_seeding, get_loaders, sample_images, test_performance

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
parser.add_argument('--test_interval', type=int, default=11, help='interval to calculate overall testing performance via L1')
opt = parser.parse_args()

generate_all_test_images = True


opt.seed_value = 12345
opt.n_epochs = 200
opt.decay_epoch= int(opt.n_epochs / 2) 
opt.img_width=64
opt.img_height=64
opt.dataset_name = 'text_segmentation'+str(opt.img_width)# 'synthtext'
opt.show_progress_every_n_iterations= 20  
opt.batch_test_size = 5
opt.AMS_grad = True
opt.sample_interval = 600
opt.test_interval = 50
opt.checkpoint_interval = 200
opt.p_RGB2BGR_augment = 0.5 # .25 # 0 indicates no change to the 
opt.p_invert_augment = 0.5 # .25
opt.aligned = False
opt.use_F1_loss = False
opt.use_white_GT = False

# Loss weights
opt.lambda_cyc = 10
opt.lambda_cycle_B = 2 # default is 1 
opt.lambda_id = 0.5 * opt.lambda_cyc
if opt.use_F1_loss:    
    opt.lambda_id_B = 10  # default is 1
else: opt.lambda_id_B = 2  # defatul is 1





''''''''''''' Main Program'''''''''''''''''''
print('\n Experiment parameters', opt, '\n') 
cuda = True if torch.cuda.is_available() else False
dataloader, val_dataloader = get_loaders(opt)
random_seeding(opt.seed_value, 
               torch.get_rng_state(), 
               torch.cuda.get_rng_state(), cuda)        

# Create sample and checkpoint directories
os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity_A = torch.nn.L1Loss()
if opt.use_F1_loss:
    criterion_identity_B = F1_loss_prime    
else:     
    criterion_identity_B = torch.nn.L1Loss()

criterion_identity_testing = torch.nn.L1Loss()

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
    if not  opt.use_F1_loss:        
        criterion_identity_A.cuda()        
        criterion_identity_B.cuda()        
        
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

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = Variable(batch['A'].type(Tensor))
        real_B = Variable(batch['B'].type(Tensor))
        
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        ''' ------------------
           Train Generators
         ------------------ '''

        optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = criterion_identity_A(G_BA(real_A), real_A)
        loss_id_B = criterion_identity_B(G_AB(real_B), real_B)
        loss_identity = loss_id_A + opt.lambda_id_B*loss_id_B

        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
        loss_GAN = loss_GAN_AB + loss_GAN_BA

        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        loss_cycle = loss_cycle_A +  opt.lambda_cycle_B*loss_cycle_B

        # Total loss
        loss_G =   ( loss_GAN + 
                    opt.lambda_cyc * loss_cycle + 
                    opt.lambda_id * loss_identity ) 

        loss_G.backward()
        optimizer_G.step()
        
        

        ''' -----------------------
           Train Discriminator A
         ----------------------- '''
         
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

        ''' -----------------------
           Train Discriminator B
         ----------------------- '''

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
        if not i % opt.show_progress_every_n_iterations:
            sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA(time-left): %s" %
                                                            (epoch, opt.n_epochs,
                                                            i, len(dataloader),
                                                            loss_D.item(), 
                                                            loss_G.item(),
                                                            loss_GAN.item(), 
                                                            loss_cycle.item(),
                                                            loss_identity.item(), time_left))
            
        # If at sample interval save image
        if not batches_done % opt.sample_interval:            
            sample_images(next(iter(val_dataloader)), batches_done, 
                          G_AB, Tensor, opt, use_max=True) # another instance


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
    if not epoch % opt.test_interval:
        test_performance(Tensor, val_dataloader, G_AB,
                                  criterion_identity_testing, use_max=False)

test_performance(Tensor, val_dataloader, G_AB,
                                  criterion_identity_testing, use_max=False)
if generate_all_test_images:
    for batch_idx, imgs in enumerate(val_dataloader):
        sample_images(imgs, batch_idx, use_max=True) # another instance
    
    
    