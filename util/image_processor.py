#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 02:05:50 2019

@author: diaa
"""
import numpy as np
import cv2
import torch

def letterbox_image(img, inp_dim):
    """
    Function:
        Resize image with unchanged aspect ratio using padding    
        
    Arguments:
        img -- image it self
        inp_dim -- dimension for resize the image (input dimension)
    
    Return:
        canvas -- resized image    
    """
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img, inp_dim):
    """
    Function:
        Prepare image for inputting to the neural network. 
        
    Arguments:
        img -- image it self
        inp_dim -- dimension for resize the image (input dimension)
    
    Return:
        img -- image after preparing 
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img
