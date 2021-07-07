"""strathan mckenzie: gnarometer"""

"""Routine for preprocessing video frames.

 Method of preprocessing is:
 -1. resize image
 -2. extract foreground
 -3. denoise image
"""


import cv2
import numpy as np


# Resize factor (downsize) for analysis:
RESIZE_FACTOR = 0.25

# Number of frames that constitute the background history:
BACKGROUND_HISTORY = 400

# Number of gaussians in BG mixture model:
NUM_GAUSSIANS = 10

# Minimum percent of frame considered background:
BACKGROUND_RATIO = 0.7

# Morphological kernel size (square):
MORPH_KERN_SIZE = 3

# Init the background modeling and foreground extraction mask.
mask = cv2.bgsegm.createBackgroundSubtractorMOG(
                                  history=BACKGROUND_HISTORY,
                                  nmixtures=NUM_GAUSSIANS,
                                  backgroundRatio=BACKGROUND_RATIO,
                                  noiseSigma=0)

# Init the morphological transformations for denoising kernel.
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                   (MORPH_KERN_SIZE, MORPH_KERN_SIZE))


def resize_image(image, factor):
    '''Resize the image to fit my laptops screen'''
    width = int(image.shape[1] * factor)
    height = int(image.shape[0] * factor)    
    dim = (width, height)   
  
    # resize image
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image


def preprocess(frame, frame_num):
    """Preprocesses video frames through resizing, background
    modeling, and denoising.

    Args:
      input: A frame from a cv2.video_reader object to process

    Returns:
      output: the preprocessed frame
    """
    #if frame_num == 119:
        #cv2.imshow('l', frame)    
    # 1. Resize the input.
    output = resize_image(frame, .6)
    lower = np.array([27,34,10], dtype="uint8")
    upper = np.array([121,132,95], dtype="uint8")
    mask = cv2.inRange(output, lower, upper)  # Find all pixels in the image within the colour range.

    
    # 2. Model the background and extract the foreground with a mask.
    #output = mask.apply(output)
    #if frame_num == 644:
        #cv2.imshow('ne', output)
        
    cv2.waitKey(0)
    # 3. Apply the morphological operators to suppress noise.
    output = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if frame_num == 300:
        print(output)
        print(frame)
        cv2.imshow('line', output)
        width = int(frame.shape[1] * 40 / 100)
        height = int(frame.shape[0] * 40 / 100)
        dim = (width, height)
        cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow('original', frame)
    cv2.waitKey(0)
    
        
    return output
