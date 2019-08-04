#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:27:13 2019

@author: malrawi

https://hardikbansal.github.io/CycleGANBlog/
https://github.com/eriklindernoren/PyTorch-GAN

Optimization
https://github.com/gkhayes/mlrose
https://towardsdatascience.com/getting-started-with-randomized-optimization-in-python-f7df46babff0

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
from F1_loss import F1_loss_torch
from misc_functions import random_seeding, get_loaders, sample_images, test_performance
import torchvision.models as torchvis_models
from torchvision.utils import save_image
import calendar

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="horse2zebra", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--batch_test_size', type=int, default=5, help='size of the test batches')
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

opt.img_height = 256
opt.img_width = 256
dt = datetime.datetime.today()
opt.dataset_name = 'text_segmentation' + str(opt.img_width) 
# opt.dataset_name = 'total_text'
#if opt.dataset_name == 'total_text':
#    opt.img_height = 512
#    opt.img_width = 512

opt.experiment_name = opt.dataset_name+'-'+ calendar.month_abbr[dt.month]+'-'+str(dt.day)
#opt.experiment_name = 'text_segmentation256-Jul-5'; opt.epoch = 450

# opt.n_residual_blocks = 16
# opt.lr= 0.00005

generate_all_test_images = True
opt.n_epochs = 1001
opt.batch_size = 1
opt.batch_test_size = 1
opt.seed_value =  12345 # np.random.randint(1, 2**32-1) 
opt.decay_epoch= 100  if opt.n_epochs>100 else opt.n_epochs//2
opt.show_progress_every_n_iterations= 20  
opt.AMS_grad = True
opt.sample_interval = 100
opt.test_interval = 10
opt.checkpoint_interval = 50
opt.p_RGB2BGR_augment = 0 # .25 # 0 means not using this augmentation
opt.p_invert_augment =  0.33 # 0 means not using this augmentation

opt.aligned = False
opt.use_F1_loss = False
opt.use_whollyG = False # use an optimizer on top of the GAN to learn the lambda's of the losses

opt.data_mode = '_prime'  # will also do affine transform wiht prob 0.3 
''' This is the background of GT one of four: 
        '': for black, 
        '_lime' for lime color, 
        '_gt': for white  text on black background, 
        '_prime': when using irrelivant synthetic text, prime is always unalighned and independent from the scene-text images 
'''

# Loss weights
#opt.lambda_cyc = 10
#opt.lambda_cycle_B = 2 # default is 1 
#opt.lambda_id = 0.5 * opt.lambda_cyc
#if opt.use_F1_loss:    
#    opt.lambda_id_B = 10  # default is 1
#else: opt.lambda_id_B = 2  # defatul is 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Flag = opt.use_whollyG # sets grad to True if using whollyG

opt.lambda_cycle_A = torch.tensor(10, dtype=torch.float32, requires_grad= Flag).to(device) # originalny 10
opt.lambda_cycle_B = torch.tensor(10, dtype=torch.float32, requires_grad=Flag).to(device) # originaly 10

opt.lambda_id_A = torch.tensor(5, dtype=torch.float32, requires_grad=Flag).to(device) # origianly 5
opt.lambda_id_B = torch.tensor(5, dtype=torch.float32, requires_grad=Flag).to(device) # originally 5

opt.lambda_GAN_AB = torch.tensor(1, dtype=torch.float32, requires_grad=Flag).to(device)  # originally 1
opt.lambda_GAN_BA = torch.tensor(1, dtype=torch.float32, requires_grad=Flag).to(device) 


''''''''''''' Main Program'''''''''''''''''''
print('\n Experiment parameters', opt, '\n') 
cuda = True if torch.cuda.is_available() else False
random_seeding(opt.seed_value, 
               torch.get_rng_state(), 
               torch.cuda.get_rng_state(), cuda)        

dataloader, val_dataloader = get_loaders(opt)
dataloader.dataset[1]

# Create sample and checkpoint directories
os.makedirs('images/%s' % opt.experiment_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.experiment_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
# criterion_GAN_B = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity_A = torch.nn.L1Loss()
criterion_identity_B = torch.nn.L1Loss()
if opt.use_F1_loss:
    criterion_identity_B = F1_loss_torch    

criterion_identity_testing = torch.nn.L1Loss() # F1_loss_torch  


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
    if not opt.use_F1_loss:        
        criterion_identity_A.cuda()        
        criterion_identity_B.cuda()        
        
if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load('saved_models/%s/G_AB_%d.pth' % (opt.experiment_name, opt.epoch)))
    G_BA.load_state_dict(torch.load('saved_models/%s/G_BA_%d.pth' % (opt.experiment_name, opt.epoch)))
    D_A.load_state_dict(torch.load('saved_models/%s/D_A_%d.pth' % (opt.experiment_name, opt.epoch)))
    D_B.load_state_dict(torch.load('saved_models/%s/D_B_%d.pth' % (opt.experiment_name, opt.epoch)))
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


##############################
#        loss optimizer Model: 
##############################        
''' Lambda's of loss optimizer 
Then, after extracting the weights from the net we have loss_G, as follows:
model_whollyG.weight  
to obtain the Lambdas    
'''
if opt.use_whollyG:
    lr_whollyG = 0.0002
    No_Lambdas = 6
    model_whollyG = torch.nn.Sequential(torch.nn.Linear(No_Lambdas, 1, bias=False)).to(device)
    with torch.no_grad():
        model_whollyG[0].weight[0, 0] = opt.lambda_cycle_B
        model_whollyG[0].weight[0, 1] = opt.lambda_GAN_AB
        model_whollyG[0].weight[0, 2] = opt.lambda_id_B
        model_whollyG[0].weight[0, 3] = opt.lambda_cycle_A        
        model_whollyG[0].weight[0, 4] = opt.lambda_id_A
        model_whollyG[0].weight[0, 5] = opt.lambda_GAN_BA
 
    optimizer_whollyG = torch.optim.Adam( 
            # model_whollyG.parameters() 
            itertools.chain( G_AB.parameters(), G_BA.parameters(), model_whollyG.parameters() )
                            ,lr=lr_whollyG, betas=(opt.b1, opt.b2), amsgrad=opt.AMS_grad
    )
    lr_scheduler_whollyG = torch.optim.lr_scheduler.LambdaLR(optimizer_whollyG, 
                lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    # loss_whollyG = torch.nn.MSELoss()
    

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

fake_A_neg_buffer = ReplayBuffer()
fake_B_neg_buffer = ReplayBuffer()


# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2**4, opt.img_width // 2**4)



# ----------
#  Training
# ----------

loss_of_id_A = []
loss_of_id_B = []

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(batch['A'].type(Tensor))
        real_B = Variable(batch['B'].type(Tensor))
        
        real_A_neg = Variable(batch['A'].type(Tensor))
        real_B_neg = Variable(batch['B'].type(Tensor))
        
            # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)
    
        ''' ------------------
           Train Generators
         ------------------ '''
        
        if opt.use_whollyG: 
            optimizer_whollyG.zero_grad() 
        else:  
            optimizer_G.zero_grad()   
    
        # Identity loss        
        loss_id_A = criterion_identity_A(G_BA(real_A), real_A)    
        loss_id_B = criterion_identity_B(G_AB(real_B), real_B)            
        
        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
        
    
        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        
        
        
        #########################  #########################
        
        if not opt.use_whollyG:
        # Total loss
            loss_G = ( 
                    opt.lambda_GAN_AB * loss_GAN_AB + 
                    opt.lambda_GAN_BA * loss_GAN_BA +                
                    opt.lambda_cycle_A * loss_cycle_A + 
                    opt.lambda_cycle_B * loss_cycle_B +
                    opt.lambda_id_A * loss_id_A +
                    opt.lambda_id_B * loss_id_B 
                    )/6 #  / (opt.lambda_GAN_AB+opt.lambda_GAN_BA+opt.lambda_cycle_B + opt.lambda_id_A+opt.lambda_id_B)
            
            loss_G.backward()
            optimizer_G.step()      
        else:            
            
            losses_in = torch.stack( # should follow the same manual weight initializgtion above                
                    (loss_cycle_B,
                     loss_GAN_AB, 
                     loss_id_B,
                     loss_cycle_A,
                     loss_id_A,
                     loss_GAN_BA)
                    )
            loss_whollyG = model_whollyG(losses_in)
            loss_whollyG = torch.abs(loss_whollyG)
            
            loss_whollyG.backward()        
            optimizer_whollyG.step()        
            loss_G =  loss_whollyG # to be used in printing the results                                           
    
            
        
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
            sys.stdout.write(
                '\r[Epoch %d/%d] [Batch %d/%d],'
                '[D loss: %f],'
                '[ G loss: %f,'
                ' adv_AB: %f,' 
                ' adv_BA:%f,' 
                ' cycle_A: %f,'
                ' cycle_B: %f,'
                ' identity_A: %f,'
                ' identity_B: %f],'            
                ' ETA(time-left): %s'           
                ' Lambda00: %.2f'
                     % (
                epoch, opt.n_epochs, i, len(dataloader),
                (loss_D_A + loss_D_B).item()/2, 
                loss_G.item(),
                loss_GAN_AB.item(),
                loss_GAN_BA.item(), 
                loss_cycle_A.item(),
                loss_cycle_B.item(),
                loss_id_A.item(), 
                loss_id_B.item(),             
                time_left,
#                model_whollyG[0].weight[0,0].cpu().item()   
                opt.lambda_id_B.item()
                
                ) )
                        
        
            
        # If at sample interval save image
        if not batches_done % opt.sample_interval:            
            sample_images(next(iter(val_dataloader)), batches_done, 
                          G_AB, Tensor, opt, use_max=True) # another instance
    
    # saving loss values, but in fact, should be averged over i
    loss_of_id_A.append(loss_id_A) 
    loss_of_id_B.append(loss_id_B)             
    
    # Update learning rates
    if opt.use_whollyG: lr_scheduler_whollyG.step(); 
        
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), 'saved_models/%s/G_AB_%d.pth' % (opt.experiment_name, epoch))
        torch.save(G_BA.state_dict(), 'saved_models/%s/G_BA_%d.pth' % (opt.experiment_name, epoch))
        torch.save(D_A.state_dict(), 'saved_models/%s/D_A_%d.pth' % (opt.experiment_name, epoch))
        torch.save(D_B.state_dict(), 'saved_models/%s/D_B_%d.pth' % (opt.experiment_name, epoch))
    if not epoch % opt.test_interval:
        test_performance(Tensor, val_dataloader, G_AB, 
                                  criterion_identity_testing)

 
    
# print('\n ................. Training the +ve vs. -ve  B classifier .............')
# test_performance = test_model_classify(criterion_classify, val_dataloader) 
test_performance(Tensor, val_dataloader, G_AB, criterion_identity_testing)
if opt.use_whollyG:
    print('Lambdas=', model_whollyG[0].weight.data.cpu().tolist()[0])
if generate_all_test_images:
    for batch_idx, imgs in enumerate(val_dataloader):
        sample_images(imgs, batch_idx, G_AB, Tensor, opt, use_max=False) # another instance
        
    
    
    
'''
            
        loss_id_A = criterion_identity_A(G_BA(real_A)+G_BA(real_A_neg), real_A+real_A_neg) 
        loss_id_B = criterion_identity_B(G_AB(real_B)+G_AB(real_B_neg), real_B+real_B_neg) 
        
        # GAN loss
        fake_B = G_AB(real_A) 
        fake_B_neg = G_AB(real_A_neg) 
        loss_GAN_AB = criterion_GAN(D_B( (fake_B+fake_B_neg)/2), valid)
        fake_A = G_BA(real_B) 
        fake_A_neg = G_BA(real_B_neg)
        loss_GAN_BA = criterion_GAN(D_A((fake_A+fake_A_neg)/2), valid)
        
    
        # Cycle loss
        recov_A = G_BA(fake_B)
        recov_A_neg =G_BA( fake_B_neg)
        loss_cycle_A = criterion_cycle((recov_A+recov_A_neg)/2, real_A)
        recov_B = G_AB(fake_A)
        recov_B_neg = G_AB(fake_A_neg)
        loss_cycle_B = criterion_cycle((recov_B+recov_B_neg)/2, real_B)
        

 
        
        #   Train Discriminator A
         
         
        optimizer_D_A.zero_grad()
    
        # Real loss
        loss_real = criterion_GAN(D_A(real_A), valid) # + criterion_GAN(D_A(real_A_neg), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        # fake_A_neg_ = fake_A_buffer.push_and_pop(fake_A_neg)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake) 
                    # + criterion_GAN(D_A(fake_A_neg_.detach()), fake)        
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2
    
        loss_D_A.backward()
        optimizer_D_A.step()
    
         
          # Train Discriminator B
         
    
        optimizer_D_B.zero_grad()
    
        # Real loss
        loss_real = criterion_GAN(D_B(real_B), valid) # + criterion_GAN(D_B(real_B_neg), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        fake_B_neg_ = fake_B_buffer.push_and_pop(fake_B_neg)
        loss_fake = criterion_GAN( D_B( (fake_B_.detach() + fake_B_neg_.detach())/2  ), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2
    
        loss_D_B.backward()
        optimizer_D_B.step()
        


'''