# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 15:02:05 2021

@author: Icarus
"""

import cv2
import numpy as np
import math
import wave_objects
import preprocess_pocket

# Boolean flag to filter blobs by inertia (shape):
FLAGS_FILTER_BY_INERTIA = True

# Minimum inertia threshold for contour:
MINIMUM_INERTIA_RATIO = 0.0
# Maximum inertia threshold for contour:
MAXIMUM_INERTIA_RATIO = 0.1

# Morphological kernel size (square):
MORPH_KERN_SIZE = 6

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
    
def filterContours(contours,
                 #area=FLAGS_FILTER_BY_AREA,
                 inertia=FLAGS_FILTER_BY_INERTIA,
                 #min_area=MINIMUM_AREA,
                 #max_area=MAXIMUM_AREA,
                 min_inertia_ratio=MINIMUM_INERTIA_RATIO,
                 max_inertia_ratio=MAXIMUM_INERTIA_RATIO):
    """Contour filtering function utilizing OpenCV.  In our case,
    we are looking for oblong shapes that exceed a user-defined area.

    Args:
      contour: A contour from an array of contours
      area: boolean flag to filter contour by area
      inertia: boolean flag to filter contour by inertia
      min_area: minimum area threshold for contour
      max_area: maximum area threshold for contour
      min_inertia_ratio: minimum inertia threshold for contour
      max_inertia_ratio: maximum inertia threshold for contour

    Returns:
      ret: A boolean TRUE if contour meets conditions, else FALSE
    """
    ret = []
    
    for contour in contours:
        # Obtain contour moments.
        moments = cv2.moments(contour)
        # Filter contours by inertia.
        if inertia is True:
            denominator = math.sqrt((2*moments['m11'])**2
                                    + (moments['m20']-moments['m02'])**2)
            epsilon = 0.01
            ratio = 0.0
    
            if denominator > epsilon:
                cosmin = (moments['m20']-moments['m02']) / denominator
                sinmin = 2*moments['m11'] / denominator
                cosmax = -cosmin
                sinmax = -sinmin
    
                imin = (0.5*(moments['m20']+moments['m02'])
                        - 0.5*(moments['m20']-moments['m02'])*cosmin
                        - moments['m11']*sinmin)
                imax = (0.5*(moments['m20']+moments['m02'])
                        - 0.5*(moments['m20']-moments['m02'])*cosmax
                        - moments['m11']*sinmax)
                ratio = imin / imax
            else:
                ratio = 1
    
            if ratio >= min_inertia_ratio or ratio < max_inertia_ratio:
                ret.append(contour)
                #center.confidence = ratio * ratio;

    return ret
    

def filterWaves(contours, contours2):
    '''words'''
    recognizedWaves = []

    for contour in contours:
    
        topmost = tuple(contour[contour[:,:,1].argmin()][0])
        bottommost = tuple(contour[contour[:,:,1].argmax()][0])
        found = False
        i = 0
        while not found and i < len(contours2):
            contour2 = contours2[i]
            topmost2 = tuple(contour2[contour2[:,:,1].argmin()][0])
            bottommost2 = tuple(contour2[contour2[:,:,1].argmax()][0])
             
            if (topmost > topmost2 and bottommost < topmost2) or (topmost2 > topmost and bottommost2 < topmost):
                recognizedWaves.append(contour)
                found = True
            i += 1
    return recognizedWaves
            
def will_be_merged(section, list_of_waves):
    """Boolean evaluating whether or not a section is in an existing
    wave's search region.

    Args:
      section: a wave object
      list_of_waves: a list of waves having search regions in which a
                     wave might fall

    Returns:
      going_to_be_merged: evaluates to True if the section is in an
                          existing wave's search region.
    """
    # All sections are initially new waves & will not be merged.
    going_to_be_merged = False

    # Find the section's major axis' projection on the y axis.
    delta_y_left = np.round(section.centroid[0]
                            * np.tan(np.deg2rad(section.axis_angle)))
    left_y = int(section.centroid[1] + delta_y_left)

    # For each existing wave, see if the section's axis falls in
    # another wave's search region.
    left = section.searchroi_coors[0][1]
    right =section.searchroi_coors[1][1]
    for wave in list_of_waves:
        wave_left = wave.searchroi_coors[0][1]
        wave_right = wave.searchroi_coors[1][1]
        if left_y >= wave.searchroi_coors[0][1] \
           and left_y <= wave.searchroi_coors[3][1]:
           #and ((wave_left <= left and wave_right >= left) or (wave_left >= left and wave_left <= right)):
            going_to_be_merged = True
            #print(wave.searchroi_coors)
            #print(9)
            #print(section.searchroi_coors)
            #print(left_y)
            break

    return going_to_be_merged
    

def find_wave(img, lower, upper, frame_num, num_frames, waves):
    # Detect a colour ball with a colour range
    mask = cv2.inRange(img, lower, upper)  # Find all pixels in the image within the colour range.
    if frame_num == 15:
            cv2.imshow('range', mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #mask = preprocess_pocket.preprocess(img, frame_num)
    #mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    #ret, mask2 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    #mask2 = cv2.bitwise_not(mask2)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #dilate = cv2.dilate(mask,kernel,iterations=3)
    #mask = cv2.blur(dilate,(15,15))
    #if len(mask2.shape) > 2:
    #    mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('threshold', mask2)
    if frame_num == 15:
            cv2.imshow('open', mask)
    # Find a series of points which outline the shape in the mask.
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours2, _hierarchy2 = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    image_copy = img.copy()

   # for contour in contours:
   #     area = cv2.contourArea(contour)
   #     hull = cv2.convexHull(contour)
   #     hull_area = cv2.contourArea(hull)
   #     solidity = float(area)/hull_area
   #     if solidity > .8:
   #         cv2.drawContours(image=image_copy, contours=contour, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    
    #remove noise
    if FLAGS_FILTER_BY_INERTIA:
        contours = filterContours(contours)
    #    contours2 = filterContours(contours2)
    #waves = filterWaves(contours, contours2)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    #contours2 = sorted(contours2, key = cv2.contourArea, reverse = True)
    #contours2 = []
    for contour in contours[0:4]:
        (x,y,w,h) = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        #print(x, y, w, h)
        #print(area)
        boxArea = (w) * (h)
        if area/boxArea > .3:
            #contours.append(contour)
            cv2.rectangle(image_copy, (x,y), (x+w,y+h), (255, 0, 0), 2)
            cv2.drawContours(image=image_copy, contours=contour, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    
   # for contour in contours2[0:4]:
   #     (x,y,w,h) = cv2.boundingRect(contour)
   #     area = cv2.contourArea(contour)
   #     #print(x, y, w, h)
   #     #print(area)
   #     boxArea = (w) * (h)
   #     if area/boxArea > .3:
   #         cv2.rectangle(image_copy, (x,y), (x+w,y+h), (100, 0, 0), 2)
   #         cv2.drawContours(image=image_copy, contours=contour, contourIdx=-1, color=(0, 100, 0), thickness=2, lineType=cv2.LINE_AA)
    
    # see the results
    cv2.imshow('None approximation', image_copy)
    

    if contours:
        sections = []
        for contour in contours[0:1]:
            #contour = max(contours, key=len)  # Assume the contour with the most points is the ball.
            # Fit a circle to the points of the contour enclosing the ball.
            (x,y),radius = cv2.minEnclosingCircle(contour)
            center = np.array([int(x),int(y)])
            radius = int(radius)
            
            image_copy_2 = img.copy()
            contours_poly = [None]*len(contours)
            boundRect = [None]*len(contours)
            centers = [None]*len(contours)
            radius = [None]*len(contours)
            for i, c in enumerate(contours[0:4]):
                contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                boundRect[i] = cv2.boundingRect(contours_poly[i])
            cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            for wave in waves:
                cv2.circle(mask, tuple(wave.centroid), wave.size//2, (0,20,100), 2)  # Draw circle around the ball.
                cv2.circle(mask, tuple(wave.centroid), 4, (0,20,40), 2)  # Draw the center (not centroid!) of the ball.
            cv2.imshow('assing', mask)
            
            #for i in range(4):
             #   color = (0, 200, 0)
                #cv2.rectangle(image_copy_2, (int(boundRect[i][0]), int(boundRect[i][1])), \
              #  cv2.drawContours(image_copy_2, contours_poly, i, color)
                 # (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)

            # Sections !!!
            section = wave_objects.Section(contour, frame_num)

            if not will_be_merged(section, waves):
                sections.append(section)
        for wave in waves:
            # Update search roi for tracking waves and merging waves.
            wave.update_searchroi_coors()
    
            # Capture all non-zero points in the new roi.
            wave.update_points(contours[0:4])
            
            #wave.update_width()
            # Check if wave has died.
            wave.update_death(frame_num)
    
            # Kill all waves if it is the last frame in the video.
            if frame_num == num_frames:
                wave.death = frame_num
    
            # Update centroids.
            wave.update_centroid()
        cv2.imshow('Contours', image_copy_2)

        
        #print(sections)         
        return sections
            #return center, radius
    else:
        return []
        #return None, None