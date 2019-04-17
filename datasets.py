import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
from PIL.ImageOps import invert as PIL_invert
import torchvision.transforms as transforms
import numpy as np
import cv2

#class Invert_B_img(object): # to be used for the inversion of B image, if needed
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
#

def color_mapping(img, Cr, Cg, Cb, ignor_zeros=False):
    #  option is to use pallet, imageWithColorPalette = colorImage.convert("P", palette=Image.ADAPTIVE, colors=8)
    # or, option to use this conversion https://realpython.com/python-opencv-color-spaces/
    
    # img.show()
    max_V= 128
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
    def __init__(self, root, transform=None, unaligned=False, mode='train', gt=False, p_color_augment=0):
        '''Args-
        gt=True, loads the white Ground Truth pixel level text, if False, use the colored  
        p_color_augment: The probability used to do color augmentation, if p is 0, no augmentaion is 
        implemented whatsoever, if p=1, then, all image instances will be augmented.
        
        '''
        self.transform = transforms.Compose(transform)
        self.mode = mode
        self.unaligned = unaligned
        self.p_color_augment = p_color_augment

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        if not gt:
            self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
        else:
            self.files_B = sorted(glob.glob(os.path.join(root, '%s/B_gt' % mode) + '/*.*'))
                       

    def __getitem__(self, index):
        item_A = Image.open(self.files_A[index % len(self.files_A)])                       

        if self.unaligned:
            item_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])            
        else:
            item_B = Image.open(self.files_B[index % len(self.files_B)]) # False gives the corresponding B to A image
        
        
        
        if self.mode == 'train':
            if np.random.rand()<self.p_color_augment:
                Cr, Cg, Cb = np.random.randint(0, 256, 3) # One coeff for each band, we will used one coeff for all            
                item_A = color_mapping(item_A, Cr, Cg, Cb)
                item_B = color_mapping(item_B, Cr, Cg, Cb, ignor_zeros=True)                
            elif np.random.rand()>0.66:
                item_A = color_mapping_cv2(item_A)
                item_B = color_mapping_cv2(item_B)                
            elif np.random.rand()>0.66:
                item_A = PIL_invert(item_A)                
                item_B = item_B.point(lambda p: 255-p if p>0 else 0 ) # invert                
                                    
        item_A_neg = self.transform(PIL_invert(item_A)) # this will only be used in validation        
        item_B = self.transform(item_B)
        item_A = self.transform(item_A)
        
        
        
        return {'A': item_A, 
                'A_neg': item_A_neg,
                'B': item_B }

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
