#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:50:50 2019

@author: malrawi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:08:08 2019

@author: malrawi
"""
import argparse
import torchvision.transforms as transforms
from datasets import ImageDataset
from torch.utils.data import DataLoader
import torch
from munit_models import Encoder #*
from F1_loss import F1_loss_numpy, F1_loss_torch
import numpy as np
import time


# from matplotlib import pyplot as plt



# A is the scene-text image
# B is the ground truth of A



parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="edges2shoes", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=128, help='size of image height')
parser.add_argument('--img_width', type=int, default=128, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between saving model checkpoints')
parser.add_argument('--n_downsample', type=int, default=2, help='number downsampling layers in encoder')
parser.add_argument('--n_residual', type=int, default=3, help='number of residual blocks in encoder / decoder')
parser.add_argument('--dim', type=int, default=64, help='number of filters in first encoder layer')
parser.add_argument('--style_dim', type=int, default=8, help='dimensionality of the style code')
opt = parser.parse_args()

def test_GAN_AB_torch(folder_model, model_name, val_dataloader, 
                double_gan=True, type_of_input_A='pos+neg'):  
    # type_of_input_A is one of these values  {'pos', 'neg', 'pos+neg', 'GAN(-nGAN+pGAN)' }
    
    
    if cuda: G_AB = G_AB.cuda()
    G_AB.load_state_dict(torch.load(folder_model + model_name ) )
    F1 = []; R=[]; P=[]; print('\n -----------------')
        
    with torch.no_grad():
        for i, imgs_batch in enumerate(val_dataloader):               
            real_B = imgs_batch['B'].to(device)                                    
            B_gan = G_AB( imgs_batch['A'].to(device) )             
            B_gan_neg = G_AB(imgs_batch['A_neg'].to(device) )
            
            f1  = F1_loss_torch( B_gan + B_gan_neg, real_B, alpha = 10000, f1_inv=False ) # needed to threshold the GT as well                                                                
            F1.append(f1.cpu().data.detach().numpy().tolist())
#            P.append(p)
#            R.append(r)
#            
        print(
                # 'R %.2f ' % (100*np.mean(R)), 
            #  'P %.2f ' % (100*np.mean(P)), 
              'F %.2f ' % (100*np.mean(F1))    
              )
            
    return np.mean(F1) # , np.mean(P), np.mean(R)



# A is the scene-text image
# B is the ground truth of A

def test_GAN_AB(folder_model, model_name, val_dataloader, threshold_value,
                double_gan=True, type_of_input_A='pos+neg'):  
    
    cuda = True if torch.cuda.is_available() else False
    if cuda: G_AB = G_AB.cuda()
    G_AB.load_state_dict(torch.load(folder_model + model_name ) )
    F1 = []; R=[]; P=[]; print('\n -----------------')
    show_img = False
    img_name_to_show = 'img_239xx.png'  #251 245 239, 
    with torch.no_grad():
        for i, imgs_batch in enumerate(val_dataloader): 
            if val_dataloader.dataset.files_A[i] == \
            '../data/text_segmentation256/train/A/'+ img_name_to_show:                
            # '../data/text_kaist_korean/test/A/'+ img_name_to_show:             
                show_img = True
                
            else: 
                show_img = False

            
            real_B = imgs_batch['B'].to(device) 
            B_gt = show_tensor(real_B) # ground truth            
            real_A = imgs_batch['A'].to(device) 
            B_gan = G_AB(real_A) 
            real_A_neg = imgs_batch['A_neg'].to(device) 
            B_gan_neg = G_AB(real_A_neg)
            if double_gan:
                B_gan =  G_AB( B_gan )
                B_gan_neg = G_AB( B_gan_neg )                    
                print('Using double GAN')  if i==0 else None
            else: print('Single GAN') if i==0 else None
            if type_of_input_A == 'pos': 
                B_GAN =  show_tensor(B_gan, show_img )
                print(' +ve A as input') if i==0 else None
            elif type_of_input_A == 'neg':
                B_GAN = show_tensor( B_gan_neg, show_img )
                print(' -ve A as input') if i==0 else None
            elif type_of_input_A == 'pos+neg' :                   
                # B_GAN = ImageChops.add_modulo(show_tensor(B_gan), show_tensor(B_gan_neg) )   
                B_GAN = show_tensor( B_gan+B_gan_neg, show_img )   
                
                print('+ve added to -ve') if i==0 else None
            elif  type_of_input_A == 'GAN(-nGAN+pGAN)': # this will override all the other options
                B_GAN = show_tensor( G_AB( G_AB(real_A)+ G_AB(real_A_neg) ),
                                    show_img)
                print('GAN(-nGAN+pGAN)') if i==0 else None
                
            
            f1, _, p, r = F1_loss_numpy(np.asarray(B_GAN)>threshold_value,#10, 
                              np.asarray(B_gt)>3) # needed to threshold the GT as well                                                                
            F1.append(f1)
            P.append(p)
            R.append(r)
            
          
        print('R %.2f ' % (100*np.mean(R)), 
              'P %.2f ' % (100*np.mean(P)), 
              'F %.2f ' % (100*np.mean(F1))    
              )
            
    return np.mean(F1), np.mean(P), np.mean(R)
        
def show_tensor(img, show_img=False):
    to_pil = transforms.ToPILImage() 
    img1  = to_pil(img.cpu().squeeze()) # we can also use test_set[1121][0].numpy()    
    if show_img: 
        img1.show()        
        img1.save('/home/malrawi/Desktop/GAN_seg_img_414/'+'gg-col'+'.png')
        img2 = img1.point(lambda p: p > 10 and 255)
        img2=img2.convert("1")
        img2.save('/home/malrawi/Desktop/GAN_seg_img_414/'+'gg-bin'+'.png')
        img2.show()
    
    return img1


dataset_name = 'text_segmentation256' 

using_test_data = True

batch_test_size = 1 
transforms_val =[ transforms.ToTensor(), 
                  transforms.Normalize((0.5,0.5,0.5), (.25,.25,.25)) 
                 ]
if using_test_data:
    val_dataloader = DataLoader(ImageDataset("../data/%s" % dataset_name, 
                                transform = transforms_val, aligned=True, 
                                mode='test', data_mode = '' ),
                                batch_size= batch_test_size, shuffle=False, num_workers=1                            
                                )

else:
    val_dataloader = DataLoader(ImageDataset("../data/%s" % dataset_name, 
                                transform = transforms_val, aligned=True, 
                                mode='train', data_mode = '' ),
                                batch_size= batch_test_size, shuffle=False, num_workers=1                            
                                )


folder_model = './munit_saved_models/text_segmentation256-Jul-10/'
model_name = 'Enc1_100.pth'
# type_of_input_A is one of these values  {'pos', 'neg', 'pos+neg', 'GAN(-nGAN+pGAN)' }

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Enc1 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Enc1.load_state_dict(torch.load(folder_model +'/%s/' % (model_name)))
print('model used', model_name)


threshold_value = 10
test_GAN_AB(folder_model, model_name, val_dataloader, threshold_value, double_gan=False, type_of_input_A='pos+neg')

