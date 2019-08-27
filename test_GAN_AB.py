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
from F1_loss import F1_loss_numpy, F1_loss_torch
import numpy as np
import time


# from matplotlib import pyplot as plt



# A is the scene-text image
# B is the ground truth of A

def test_GAN_AB_torch(folder_model, model_name, val_dataloader, 
                double_gan=True, type_of_input_A='pos+neg'):  
    # type_of_input_A is one of these values  {'pos', 'neg', 'pos+neg', 'GAN(-nGAN+pGAN)' }
    n_residual_blocks = 9 # this should be the same values used in training the G_AB model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G_AB = GeneratorResNet(res_blocks=n_residual_blocks)
    cuda = True if torch.cuda.is_available() else False
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
    # type_of_input_A is one of these values  {'pos', 'neg', 'pos+neg', 'GAN(-nGAN+pGAN)' }
    n_residual_blocks = 9 # this should be the same values used in training the G_AB model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G_AB = GeneratorResNet(res_blocks=n_residual_blocks)
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
# dataset_name = 'total_text'     
# dataset_name = 'text_kaist_korean'

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

# folder_model = './saved_models/text_segmentation512-May-30/'
#folder_model = './saved_models/aligned-text_segmentation256-may25/'
# folder_model = './saved_models/unaligned-text_segmentation256-may26/'
folder_model ='./saved_models/text_segmentation256-Jun-2/'
# folder_model = './saved_models/total_text-Jun-4/'
# folder_model ='./saved_models/text_segmentation256-Jun-10/' # usin new lambda set
# folder_model ='./saved_models/text_segmentation256-Jun-11/'  # using the newest lambda set
# folder_model ='./saved_models/text_segmentation256-Jun-12/'  # using the newest lambda set
# folder_model ='./saved_models/text_segmentation256-Jun-30/'
# folder_model ='./saved_models/text_segmentation256-May-27-Bprime-color/'
# folder_model = './saved_models/text_segmentation256-Jul-10/'


model_name = 'G_AB_300.pth'
print('model used', model_name)

print('Using Torch based code to find F1')
test_GAN_AB_torch(folder_model, model_name, val_dataloader, 
                  double_gan=False, type_of_input_A='pos+neg')
print('Using Numpy based code to find F1')
threshold_value = 10
test_GAN_AB(folder_model, model_name, val_dataloader, threshold_value, double_gan=False, type_of_input_A='pos')
test_GAN_AB(folder_model, model_name, val_dataloader, threshold_value, double_gan=False, type_of_input_A='neg')
test_GAN_AB(folder_model, model_name, val_dataloader, threshold_value, double_gan=False, type_of_input_A='pos+neg')
test_GAN_AB(folder_model, model_name, val_dataloader, threshold_value,  double_gan=True, type_of_input_A='pos')
test_GAN_AB(folder_model, model_name, val_dataloader, threshold_value, double_gan=True, type_of_input_A='neg')
test_GAN_AB(folder_model, model_name, val_dataloader,threshold_value, double_gan=True, type_of_input_A='pos+neg')
test_GAN_AB(folder_model, model_name, val_dataloader, threshold_value, double_gan=True, type_of_input_A='GAN(-nGAN+pGAN)' )


# x= [ i for i in range(len(f1))]
# plt.scatter(x, f1)



#            p_time= time.time()
#            for i in range(1000):
#                real_A = imgs_batch['A'].to(device) 
#                B_gan = G_AB(real_A) 
#                
#            print(time.time()-p_time)