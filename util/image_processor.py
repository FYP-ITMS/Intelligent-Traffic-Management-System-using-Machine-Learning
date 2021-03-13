import numpy as np
import cv2
import torch

def letterbox_image(image, input_dimension):
    """
    Function:
        Resize image with unchanged aspect ratio using padding    
        
    Arguments:
        image -- image input passed.
        input_dimension -- dimensions for resizing the image.
    
    Return:
        image_as_tensor -- resized image    
    """
    image_width, image_height = image.shape[1], image.shape[0]
    width, height = input_dimension
    new_width = int(image_width * min(width/image_width, height/image_height))
    new_height = int(image_height * min(width/image_width, height/image_height))
    resized_image = cv2.resize(image, (new_width,new_height), interpolation = cv2.INTER_CUBIC)
    
    image_as_tensor = np.full((input_dimension[1], input_dimension[0], 3), 128)
    image_as_tensor[(height-new_height)//2:(height-new_height)//2 + new_height,(width-new_width)//2:(width-new_width)//2 + new_width,  :] = resized_image
    
    return image_as_tensor

def preparing_image(image, input_dimension):
    """
    Function:
        Prepare image for inputting to the neural network. 
        
    Arguments:
        age input passed.
        input_dimension -- dimensions for resizing the image.
    
    Return:
        image -- image after preparing 
    """
    image = (letterbox_image(image, (input_dimension, input_dimension)))
    image = image[:,:,::-1].transpose((2,0,1)).copy()
    image = torch.from_numpy(image).float().div(255.0).unsqueeze(0)

    return image
