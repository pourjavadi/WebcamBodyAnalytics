# WebcamBodyAnalytics

A real-time computer vision project that analyzes 40 human body and environmental features using a standard webcam. It leverages Mediapipe, YOLOv8, OpenCV, and Scipy to track physiological, motion, facial, and environmental data, such as heart rate, breathing rate, facial expressions, eye gaze, and object detection.

## Features
This project supports the following real-time analyses:
1. Heart rate estimation (via Photoplethysmography)
2. Breathing rate detection (chest movement)
3. Joint angle measurement (e.g., elbow)
4. Facial expression recognition (smile, frown, neutral)
5. Eye gaze tracking
6. Blink detection
7. Skin color change analysis
8. Body tremor detection
9. Head movement analysis
10. Hand gesture recognition (OK, Victory, Heart, Fist)
11. Object detection in the environment (using YOLOv8)
12-40. Additional features (placeholders for walking pattern, wrist pulse, sweating, diaphragm movement, muscle fatigue, body balance, eye movement, pupil size, eyebrow movement, face symmetry, lip movement, neck angle, chest movement, eye distance, wrist movement, leg movement, facial tics, ear movement, mouth opening speed, thumb movement, waist angle, finger movement, lip color, cheek movement, full body movement, hand distance)

Data is displayed on the video feed and saved in a CSV file for further analysis.

## Prerequisites
- **Python**: 3.10
- **Required packages**:
  ```bash
  pip install opencv-python mediapipe pandas ultralytics scipy numpy
  ```

- **Webcam**: Minimum 720p resolution recommended for accurate tracking of facial features, heart rate, skin color changes, and motion analysis. A 1080p webcam improves precision for fine-grained features like pupil size, lip color, or ear movement. Ensure the webcam is properly connected and supports at least 15 FPS for stable real-time processing.
  
- **Operating System**: Compatible with Windows (tested on Windows 10/11), macOS (tested on macOS Ventura), and Linux (tested on Ubuntu 20.04/22.04). Some features, like YOLOv8, may perform better on Linux with GPU support.
  
- **Hardware**: A mid-range CPU (e.g., Intel i5/i7 or AMD Ryzen 5, 4 cores or more) and 8GB RAM are recommended for smooth performance at ~20 FPS. For optimal performance, especially with YOLOv8 object detection, a GPU (e.g., NVIDIA GTX 1060 or better) is highly recommended, though the project runs on CPU alone. At least 2GB of free disk space is needed for storing snapshots, videos, and CSV data.
  
- **Lighting**: Stable, bright, white lighting (e.g., LED or natural daylight) is critical for accurate heart rate estimation, skin color analysis, and facial feature detection. Avoid flickering lights (e.g., fluorescent) or heavy shadows, as they can introduce noise in PPG signals or facial tracking.

## Controls
- `q`: Quit the program.
- `s`: Save a snapshot to `snapshots/snapshot_YYYYMMDD_HHMMSS.jpg`.
- `r`: Start/stop video recording to `videos/video_YYYYMMDD_HHMMSS.mp4`.

### Outputs
- Real-time metrics displayed on the video feed, including heart rate (BPM), breathing rate (BPM), elbow angle (degrees), facial expression (smile, frown, neutral), eye gaze coordinates (x, y), blink status (Yes/No), skin color intensity (red channel), body tremor level (variance), head movement (variance), hand gestures (OK, Victory, Heart, Fist), and detected objects (e.g., "chair 0.85").
- Data saved in `landmark_data/data_YYYYMMDD_HHMMSS.csv` with columns for timestamp, heart_rate, breathing_rate, elbow_angle, facial_expression, gaze_x, gaze_y, blink, skin_color, tremor, head_movement, and placeholders for all 40 features.
- Logs saved in `detection_log.txt` for debugging (e.g., "Not enough peaks for heart rate" or "Invalid forehead region").

## Limitations
- Placeholder features (e.g., pupil size, lip color, wrist pulse, ear movement) return default values (0 or None) and need advanced algorithms or high-resolution webcams for full implementation.
- Heart rate and breathing rate accuracy depend on lighting and webcam quality. Poor conditions (e.g., dim light, low resolution) may cause unstable or N/A readings.
- High computational load (Mediapipe + YOLO) may reduce FPS on low-end hardware (<10 FPS on older CPUs). A GPU or optimized settings can help.
- Features like sweating, facial t
