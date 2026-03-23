# wave_tracker
A computer vision project that detects and tracks surfable waves from video footage using Python, NumPy and OpenCV.

The system processes raw surf footage frame-by-frame, applying image processing techniques to identify wave motion and highlight rideable waves in real time.

## Steps
The application follows a multi-step computer vision pipeline:

1. Frame Extraction

- Video is read frame-by-frame using OpenCV

- Each frame is converted into a format suitable for processing

2. Preprocessing

- Convert frames to grayscale to simplify computation

- Apply Gaussian blur to reduce noise and smooth the image

- Improve signal-to-noise ratio for motion detection

3. Motion Detection

- Compute frame differences or background subtraction

- Identify areas of change between consecutive frames

- Highlight moving regions (wave motion)

4. Thresholding

- Apply binary thresholding to isolate significant motion

- Filter out small/noisy changes (e.g. ripples, camera noise)

5. Contour Detection

- Detect contours in the thresholded image

- Identify candidate wave shapes based on size and structure

6. Wave Tracking

- Track detected contours across frames

- Maintain consistency of wave movement over time

- Filter out short-lived or irrelevant detections

7. Visualisation

- Draw bounding boxes or contours around detected waves

- Overlay tracking information onto the original frame

8. Output Rendering

- Processed frames are written to an output video file

- Final result saved as Tracked.mp4

To run:

run find_wave.py, specifying the video file to analyse, stored in the Videos folder. Resulting video is written to Tracked.mp4.

e.g. python find_wave.py <video_filename>

## notes
- Works best with stable camera footage (e.g. tripod or drone)
- Performance depends on lighting, wave size, and video quality
- This is a heuristic-based approach (not ML-based)

## Future Improvements
- Use optical flow for more accurate motion tracking
- Apply machine learning for wave classification
- Improve robustness to lighting and camera movement
- Real-time processing with live video input
