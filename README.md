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
- Python 3.10
- Required packages:
  ```bash
  pip install opencv-python mediapipe pandas ultralytics scipy numpy
