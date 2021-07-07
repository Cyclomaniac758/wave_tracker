# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 12:12:30 2021

@author: Icarus
"""

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
import os
import time
import find_wave
import sys
import write_report



# Resize factor (downsize) for analysis:
RESIZE_FACTOR = .6

# Number of frames that constitute the background history:
BACKGROUND_HISTORY = 400

# Number of gaussians in BG mixture model:
NUM_GAUSSIANS = 5

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

def status_update(frame_number, tot_frames):
    """A simple inline status update for stdout.
    Prints frame number for every 100 frames completed.

    Args:
      frame_number: number of frames completed
      tot_frames: total number of frames to analyze

    Returns:
      VOID: writes status to stdout
    """
    if frame_number == 1:
        sys.stdout.write("Starting analysis of %d frames...\n" %tot_frames)
        sys.stdout.flush() 

    if frame_number % 100 == 0:
        sys.stdout.write("%d" %frame_number)
        sys.stdout.flush()
    elif frame_number % 10 == 0:
        sys.stdout.write(".")
        sys.stdout.flush()

    if frame_number == tot_frames:
        print("End of video reached successfully.")
        
def create_video_writer(input_video):
    """Creates a OpenCV Video Writer object using the mp4c codec and
    input video stats (frame width, height, fps) for tracking
    visualization.

    Args:
      input_video: video read into program using opencv methods

    Returns:
      out: cv2 videowriter object to which individual frames can be
           written
    """
    # Grab some video stats for videowriter object.
    original_width = input_video.get(cv2.CAP_PROP_FRAME_WIDTH)
    original_height = input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = input_video.get(cv2.CAP_PROP_FPS)

    # Initiate video writer object by defining the codec and initiating
    # the VideoWriter object.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join("tracked.mp4")
    out = cv2.VideoWriter(output_path,
                          fourcc,
                          fps,
                          (int(original_width), int(original_height)),
                          isColor=True)

    return out



def track_wave_with_kalman():
    # cap = cv2.VideoCapture('../images/red_ball_roll.avi')
    cap = cv2.VideoCapture('videos/laniake_test.mp4')
    out = create_video_writer(cap)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 1
    timestep = 1/25  # Time between frames in the video.
    # Define the upper and lower colour thresholds for the ball colour.
    #lower = np.array([30,30,0], dtype="uint8")
    #upper = np.array([100,100, 35], dtype="uint8")
    ## Good for laniake
    lower = np.array([27,34,10], dtype="uint8")
    upper = np.array([121,132,95], dtype="uint8")
    ## Experimenting for Uluwatu, good for uluwatu2
    #lower = np.array([0,0,0], dtype="uint8")
    #upper = np.array([121,132,40], dtype="uint8")
    waves = []
    time_start = time.time()
    done_waves = []
    
    while cap.isOpened():
        status_update(frame_number, num_frames)

        ret, frame = cap.read()  # Read an frame from the video file.
        if not ret: # If we cannot read any more frames from the video file, then exit.
            break
        #frame = mwt_preprocessing.preprocess(frame, 0)
        #cv2.imshow("ye", frame)
        frame = find_wave.resize_image(frame, RESIZE_FACTOR)
        #center, radius = find_wave.find_wave(frame, lower, upper, frame_number)  # Search for the wave in the frame.
        new_waves = find_wave.find_wave(frame, lower, upper, frame_number, num_frames, waves)  # Search for the wave in the frame.
        #print('\nMeasurement:\t', center)
        
        dead_recognized_waves = [wave for wave in waves 
                                 if wave.death is not None
                                 and wave.recognized is True]
        done_waves = done_waves + [wave for wave in waves if wave.death is not None]
        waves = [wave for wave in waves if wave.death is None]
        waves = waves + new_waves
        #print(len(waves))
        for wave in waves:
            kalman = wave.getKalman()
            kalman.predict()
            
            center_ = (int(kalman.x[0]), int(kalman.x[1]))
            axis_lengths = (int(kalman.P_prior[0, 0]), int(kalman.P_prior[1, 1]))
            cv2.ellipse(frame, center_, axis_lengths, 0, 0, 360, color=(255, 0, 0))
        
            
            cv2.circle(frame, tuple(wave.centroid), wave.size//2, (0,255,0), 2)  # Draw circle around the ball.
            cv2.circle(frame, tuple(wave.centroid), 4, (0,255,0), 2)  # Draw the center (not centroid!) of the ball.
    
            # The Kalman filter expects the x,y coordinates in a 2D array.
            measured = np.array([wave.centroid[0], wave.centroid[1]], dtype="float32")
            # Update the Kalman filter with the waves location if we have it.
            kalman.update(measured)
            wave.setKalman(kalman)

            #print('Estimate:\t', np.int32(kalman.x))
        cv2.imshow('frame1', frame)  # Display the grayscale frame on the screen.

  
        # resize image
        frame = find_wave.resize_image(frame, 1/RESIZE_FACTOR)
        cv2.imshow('o', frame)
        out.write(frame)
        frame_number += 1

        if cv2.waitKey(80) & 0xFF == ord('q'):
            break

    # Release the video file, and close the GUI.
    time_elapsed = (time.time() - time_start)
    performance = (num_frames / time_elapsed)
    write_report.write_report(done_waves, performance)
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #track_wave()
    track_wave_with_kalman()
    #track_wave_with_unscented_kalman()