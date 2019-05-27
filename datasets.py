import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
from PIL.ImageOps import invert as PIL_invert
import torchvision.transforms as transforms
import numpy as np
import cv2

def color_mapping(img, Cr, Cg, Cb, ignor_zeros=True):
    #  option is to use pallet, imageWithColorPalette = colorImage.convert("P", palette=Image.ADAPTIVE, colors=8)
    # or, option to use this conversion https://realpython.com/python-opencv-color-spaces/
    
    # img.show()
    max_V= 256
    red, green, blue = img.split()    
    if not ignor_zeros:
        red = red.point(lambda p: (p+Cr)%max_V) # when Cr=0, this will give the negative(invert/complement) image    
        green = green.point(lambda p: (p+Cg)%max_V) # when Cr=0, this will give the negative(invert/complement) image
        blue = blue.point(lambda p: (p+Cb)%max_V) # when Cr=0, this will give the negative(invert/complement) image    
    else: 
        red = red.point(lambda p: (p+Cr)%max_V if p>200  else 0) # when Cr=0, this will give the negative(invert/complement) image    
        green = green.point(lambda p: (p+Cg)%max_V if p>200  else 0) # when Cr=0, this will give the negative(invert/complement) image
        blue = blue.point(lambda p: (p+Cb)%max_V if p>200  else 0) # when Cr=0, this will give the negative(invert/complement) image    
    
    img = Image.merge("RGB", (red, green, blue))
    # img.show()
    
    return img


''' https://docs.opencv.org/2.4.13.7/modules/contrib/doc/facerec/colormaps.html
https://www.programcreek.com/python/example/71304/cv2.COLOR_BGR2RGB 
'''
def color_mapping_cv2(img):
    img = np.asarray(img)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)
    
    return img
    


class ImageDataset(Dataset):
    def __init__(self, root, transform=None, aligned=True, mode='train', 
                 data_mode = '', p_color_augment=0, p_RGB2BGR_augment=0, 
                 p_invert_augment=0):
        '''Args-
        gt=True, loads the white Ground Truth pixel level text, if False, use the colored  
        p_color_augment: The probability used to do color augmentation, if p is 0, no augmentaion is 
        implemented whatsoever, if p=1, then, all image instances will be augmented.
        
        '''
        self.transform = transforms.Compose(transform)
        self.mode = mode
        self.aligned = aligned
        self.p_color_augment = p_color_augment
        self.p_RGB2BGR_augment = p_RGB2BGR_augment
        self.p_invert_augment = p_invert_augment        
        self.data_mode = data_mode
        folder_name = '%s/B'+ data_mode
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, folder_name % mode) + '/*.*'))
        sheer_tsfm = transforms.RandomAffine(degrees =(-20, 20), shear=(-30, 30) )         
        self.random_sheer = transforms.Compose(
                [transforms.RandomApply([sheer_tsfm], p = 0.3)]) # will only be used if cf.use_distortion_augmentor is True
        
    def __getitem__(self, index):
        item_A = Image.open(self.files_A[index % len(self.files_A)])                                
        if self.aligned and not self.data_mode=='B_prime'  :
            item_B = Image.open(self.files_B[index % len(self.files_B)]) # False gives the corresponding B to A image                        
        else:            
            item_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]) 
            if self.data_mode=='B_prime': item_B = self.random_sheer(item_B)
        
        if self.mode == 'train' and self.data_mode != 'lime' : # we don't want lime to be corrupted
            if np.random.rand() < self.p_RGB2BGR_augment: # will not run when p_RGB2BGR_augment=0
                item_A = color_mapping_cv2(item_A)                
                item_B = color_mapping_cv2(item_B)                                                                        
            if np.random.rand() < self.p_invert_augment: # will not run when p_invert_augment=0
                item_A = PIL_invert(item_A)                  
                if not self.data_mode=='B_prime' : # we do not want to invert B_prime
                    item_B = item_B.point(lambda p: 255-p if p>0 else 0 ) # invert                                
        
        if self.data_mode =='gt': item_B_neg = item_B # no need to invert gt(white text) here
        else: item_B_neg = item_B.point(lambda p: 255-p if p>0 else 0 ) # invert                                
        
        # transformation
        item_A_neg = self.transform(PIL_invert(item_A)) # this will only be used in validation                
        item_B = self.transform(item_B)
        item_A = self.transform(item_A) 
        item_B_neg= self.transform(item_B_neg)
        
        return {'A': item_A, 
                'A_neg': item_A_neg,
                'B': item_B,
                'B_neg': item_B_neg}

    def __len__(self):
        # return min(len(self.files_A), len(self.files_B)) # original
         return min(len(self.files_A), len(self.files_B))
#        if self.mode == 'train' and self.data_mode =='B_prime':
#            return min(len(self.files_A), len(self.files_B)) # since B_prime has lots of B samples, but A samples are the training samples, and we are randomly sampling from B_prime
#        else:  max(len(self.files_A), len(self.files_B))




#elif np.random.rand() < self.p_color_augment: # will not run when p_color_augment=0
#                # Cr, Cg, Cb = np.random.randint(0, 256, 3) # One coeff for each band, we will used one coeff for all            
#               
#                Cr, Cg, Cb = (32, 32, 32)  # middle point inversion
#                item_A = color_mapping(item_A, Cr, Cg, Cb)
#                item_B = color_mapping(item_B, Cr, Cg, Cb, ignor_zeros=True)                
#
#
#class NoneTransform(object):
#    """ Does nothing to the image, to be used instead of None    
#    Args:
#        image in, image out, nothing is done
#    """
#        
#    def __call__(self, image):       
#        return image
#
#class Invert_B_img(object):
#    """ Does nothing to the image, to be used instead of None    
#    Args:
#        image in, image out, nothing is done
#    """
#        
#    def __call__(self, image):                         
#        image = image.point(lambda p: 255-p if p>8 and p<64 else 0 ) # invert         
##        
#        
#        return image
