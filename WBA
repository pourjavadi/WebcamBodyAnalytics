import cv2
import mediapipe as mp
import numpy as np
import time
import os
from datetime import datetime
import csv
import pandas as pd
import logging
from ultralytics import YOLO
from scipy import signal

# Configure logging
logging.basicConfig(filename='detection_log.txt', level=logging.INFO, 
                   format='%(asctime)s - %(message)s')

# Setup Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Function to count fingers
def count_fingers(hand_landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    count = 0
    if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_tips[0] - 1].x:
        count += 1
    for tip in finger_tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    return count

# Function to detect hand gestures
def detect_gesture(hand_landmarks):
    landmarks = hand_landmarks.landmark
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    if distance < 0.05 and count_fingers(hand_landmarks) == 2:
        return "OK"
    if count_fingers(hand_landmarks) == 2 and landmarks[8].y < landmarks[12].y:
        return "Victory"
    if count_fingers(hand_landmarks) == 2 and landmarks[8].x < landmarks[12].x:
        return "Heart"
    if count_fingers(hand_landmarks) == 0:
        return "Fist"
    return "None"

# Function to calculate heart rate
def calculate_heart_rate(green_signal, timestamps, fps):
    if len(green_signal) < 50 or len(timestamps) < 50:
        return None
    green_signal = np.array(green_signal)
    timestamps = np.array(timestamps)
    if len(green_signal) != len(timestamps):
        return None
    try:
        green_signal = (green_signal - np.mean(green_signal)) / (np.std(green_signal) + 1e-10)
        b, a = signal.butter(4, [0.7 / (fps / 2), 2.0 / (fps / 2)], btype='band')
        filtered = signal.filtfilt(b, a, green_signal)
        peaks, _ = signal.find_peaks(filtered, distance=int(fps / 2), height=np.std(filtered) * 0.5)
        if len(peaks) < 2:
            return None
        heart_rate = 60 / np.mean(np.diff(timestamps[peaks]))
        if 40 < heart_rate < 180:
            return heart_rate
        return None
    except:
        return None

# Function to calculate breathing rate
def calculate_breathing_rate(pose_landmarks, chest_positions, timestamps, fps):
    if not pose_landmarks or len(chest_positions) < 50:
        return None
    chest_y = np.array([pos[1] for pos in chest_positions])
    timestamps = np.array(timestamps[-len(chest_positions):])
    try:
        b, a = signal.butter(4, [0.1 / (fps / 2), 0.5 / (fps / 2)], btype='band')
        filtered = signal.filtfilt(b, a, chest_y)
        peaks, _ = signal.find_peaks(filtered, distance=int(fps / 2))
        if len(peaks) < 2:
            return None
        breathing_rate = 60 / np.mean(np.diff(timestamps[peaks]))
        if 5 < breathing_rate < 30:
            return breathing_rate
        return None
    except:
        return None

# Function to calculate joint angle
def calculate_joint_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Function to detect facial expression
def detect_facial_expression(face_landmarks):
    if not face_landmarks:
        return "None"
    mouth_left = face_landmarks.landmark[61]
    mouth_right = face_landmarks.landmark[291]
    mouth_top = face_landmarks.landmark[0]
    mouth_bottom = face_landmarks.landmark[17]
    mouth_width = np.sqrt((mouth_right.x - mouth_left.x)**2 + (mouth_right.y - mouth_left.y)**2)
    mouth_height = np.sqrt((mouth_top.y - mouth_bottom.y)**2)
    if mouth_width / mouth_height > 2.0:
        return "Smile"
    if mouth_height / mouth_width > 1.5:
        return "Frown"
    return "Neutral"

# Function to track eye gaze
def track_eye_gaze(face_landmarks):
    if not face_landmarks:
        return None
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]
    gaze_x = (left_eye.x + right_eye.x) / 2
    gaze_y = (left_eye.y + right_eye.y) / 2
    return gaze_x, gaze_y

# Function to detect blink
def detect_blink(face_landmarks, blink_history):
    if not face_landmarks:
        return False
    left_eye_top = face_landmarks.landmark[159]
    left_eye_bottom = face_landmarks.landmark[145]
    eye_distance = np.sqrt((left_eye_top.y - left_eye_bottom.y)**2)
    blink = eye_distance < 0.015
    blink_history.append(blink)
    if len(blink_history) > 3 and sum(blink_history[-3:]) >= 2:
        return True
    return False

# Function to detect skin color change
def detect_skin_color_change(face_region):
    if face_region.size == 0:
        return None
    red_channel = np.mean(face_region[:, :, 2])
    return red_channel

# Function to detect body tremor
def detect_body_tremor(pose_landmarks, tremor_history):
    if not pose_landmarks:
        return 0
    shoulder = pose_landmarks.landmark[12]
    tremor_history.append([shoulder.x, shoulder.y])
    if len(tremor_history) < 10:
        return 0
    positions = np.array(tremor_history[-10:])
    tremor = np.std(np.diff(positions, axis=0))
    return tremor

# Function to analyze head movement
def analyze_head_movement(face_landmarks, head_history):
    if not face_landmarks:
        return 0
    head_pos = face_landmarks.landmark[10]
    head_history.append([head_pos.x, head_pos.y])
    if len(head_history) < 10:
        return 0
    return np.std(np.diff(np.array(head_history[-10:]), axis=0))

# Placeholder functions for remaining features (to be expanded)
def analyze_walking_pattern(pose_landmarks):
    return 0  # Placeholder for walking pattern analysis

def detect_pulse_wrist(hand_landmarks):
    return None  # Placeholder for wrist pulse detection

def detect_sweating(face_region):
    return None  # Placeholder for sweating analysis

def detect_diaphragm_movement(pose_landmarks):
    return 0  # Placeholder for diaphragm movement

def detect_muscle_fatigue(pose_landmarks):
    return 0  # Placeholder for muscle fatigue

def analyze_body_balance(pose_landmarks):
    return 0  # Placeholder for body balance

def analyze_eye_movement(face_landmarks):
    return 0  # Placeholder for eye movement speed

def detect_pupil_size(face_landmarks):
    return None  # Placeholder for pupil size

def analyze_eyebrow_movement(face_landmarks):
    return 0  # Placeholder for eyebrow movement

def detect_face_symmetry(face_landmarks):
    return 0  # Placeholder for face symmetry

def detect_lip_movement(face_landmarks):
    return 0  # Placeholder for lip movement

def analyze_neck_angle(pose_landmarks):
    return 0  # Placeholder for neck angle

def analyze_chest_movement(pose_landmarks):
    return 0  # Placeholder for chest movement

def detect_eye_distance(face_landmarks):
    return 0  # Placeholder for eye distance

def analyze_wrist_movement(hand_landmarks):
    return 0  # Placeholder for wrist movement

def analyze_leg_movement(pose_landmarks):
    return 0  # Placeholder for leg movement

def detect_facial_tics(face_landmarks):
    return 0  # Placeholder for facial tics

def detect_ear_movement(face_landmarks):
    return 0  # Placeholder for ear movement

def analyze_mouth_opening_speed(face_landmarks):
    return 0  # Placeholder for mouth opening speed

def analyze_thumb_movement(hand_landmarks):
    return 0  # Placeholder for thumb movement

def analyze_waist_angle(pose_landmarks):
    return 0  # Placeholder for waist angle

def analyze_finger_movement(hand_landmarks):
    return 0  # Placeholder for finger movement

def detect_lip_color(face_region):
    return None  # Placeholder for lip color

def analyze_cheek_movement(face_landmarks):
    return 0  # Placeholder for cheek movement

def analyze_full_body_movement(pose_landmarks):
    return 0  # Placeholder for full body movement

def analyze_hand_distance(pose_landmarks):
    return 0  # Placeholder for hand distance

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables
prev_time = 0
save_dir = "snapshots"
video_dir = "videos"
data_dir = "landmark_data"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

recording = False
out = None
gesture_counter = {"OK": 0, "Victory": 0, "Heart": 0, "Fist": 0, "None": 0}
green_signal = []
timestamps = []
chest_positions = []
blink_history = []
tremor_history = []
head_history = []
heart_rate_buffer = []
HEART_RATE_SMOOTHING_FACTOR = 0.7

# CSV file
csv_file = os.path.join(data_dir, f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
csv_columns = [
    'timestamp', 'heart_rate', 'breathing_rate', 'body_speed', 'elbow_angle', 'facial_expression',
    'gaze_x', 'gaze_y', 'blink', 'skin_color', 'tremor', 'head_movement', 'walking_pattern',
    'pulse_wrist', 'sweating', 'diaphragm_movement', 'muscle_fatigue', 'body_balance',
    'eye_movement', 'pupil_size', 'eyebrow_movement', 'face_symmetry', 'lip_movement',
    'neck_angle', 'chest_movement', 'eye_distance', 'wrist_movement', 'leg_movement',
    'facial_tics', 'ear_movement', 'mouth_opening_speed', 'thumb_movement', 'waist_angle',
    'finger_movement', 'lip_color', 'cheek_movement', 'full_body_movement', 'hand_distance'
]
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_columns)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with Holistic
    results = holistic.process(rgb_frame)
    
    # Process with YOLO
    yolo_results = yolo_model(rgb_frame, conf=0.5)
    
    # Face analysis
    face_landmark_count = 0
    face_data = []
    face_region = np.zeros((1, 1, 3), dtype=np.uint8)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)
        )
        face_landmark_count = len(results.face_landmarks.landmark)
        face_data = [(lm.x, lm.y) for lm in results.face_landmarks.landmark]
        
        # Heart rate (forehead)
        ih, iw, _ = frame.shape
        forehead = results.face_landmarks.landmark[10]
        fx, fy = int(forehead.x * iw), int(forehead.y * ih)
        x_min = max(fx - 20, 0)
        x_max = min(fx + 20, iw)
        y_min = max(fy - 20, 0)
        y_max = min(fy + 20, ih)
        face_region = frame[y_min:y_max, x_min:x_max]
        if face_region.size > 0:
            green_signal.append(np.mean(face_region[:, :, 1]))
            timestamps.append(curr_time)
    
    # Limit buffer
    if len(green_signal) > 200:
        green_signal = green_signal[-200:]
        timestamps = timestamps[-200:]
    
    # Calculate heart rate
    heart_rate = calculate_heart_rate(green_signal, timestamps, fps)
    if heart_rate is not None:
        if heart_rate_buffer:
            heart_rate = HEART_RATE_SMOOTHING_FACTOR * heart_rate + (1 - HEART_RATE_SMOOTHING_FACTOR) * heart_rate_buffer[-1]
        heart_rate_buffer.append(heart_rate)
        if len(heart_rate_buffer) > 10:
            heart_rate_buffer = heart_rate_buffer[-10:]
        heart_rate = np.mean(heart_rate_buffer)
    
    # Body analysis
    pose_landmark_count = 0
    pose_data = []
    body_speed = 0
    elbow_angle = 0
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        pose_landmark_count = len(results.pose_landmarks.landmark)
        pose_data = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
        
        # Body speed
        chest = results.pose_landmarks.landmark[12]
        chest_positions.append([chest.x * iw, chest.y * ih])
        if len(chest_positions) > 200:
            chest_positions = chest_positions[-200:]
        
        # Elbow angle
        shoulder = np.array([results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[11].y])
        elbow = np.array([results.pose_landmarks.landmark[13].x, results.pose_landmarks.landmark[13].y])
        wrist = np.array([results.pose_landmarks.landmark[15].x, results.pose_landmarks.landmark[15].y])
        elbow_angle = calculate_joint_angle(shoulder, elbow, wrist)
    
    # Breathing rate
    breathing_rate = calculate_breathing_rate(results.pose_landmarks, chest_positions, timestamps, fps)
    
    # Facial analysis
    facial_expression = detect_facial_expression(results.face_landmarks)
    gaze = track_eye_gaze(results.face_landmarks)
    blink = detect_blink(results.face_landmarks, blink_history)
    skin_color = detect_skin_color_change(face_region)
    head_movement = analyze_head_movement(results.face_landmarks, head_history)
    
    # Body tremor
    tremor = detect_body_tremor(results.pose_landmarks, tremor_history)
    
    # Placeholder feature calculations
    walking_pattern = analyze_walking_pattern(results.pose_landmarks)
    pulse_wrist = detect_pulse_wrist(results.left_hand_landmarks)
    sweating = detect_sweating(face_region)
    diaphragm_movement = detect_diaphragm_movement(results.pose_landmarks)
    muscle_fatigue = detect_muscle_fatigue(results.pose_landmarks)
    body_balance = analyze_body_balance(results.pose_landmarks)
    eye_movement = analyze_eye_movement(results.face_landmarks)
    pupil_size = detect_pupil_size(results.face_landmarks)
    eyebrow_movement = analyze_eyebrow_movement(results.face_landmarks)
    face_symmetry = detect_face_symmetry(results.face_landmarks)
    lip_movement = detect_lip_movement(results.face_landmarks)
    neck_angle = analyze_neck_angle(results.pose_landmarks)
    chest_movement = analyze_chest_movement(results.pose_landmarks)
    eye_distance = detect_eye_distance(results.face_landmarks)
    wrist_movement = analyze_wrist_movement(results.left_hand_landmarks)
    leg_movement = analyze_leg_movement(results.pose_landmarks)
    facial_tics = detect_facial_tics(results.face_landmarks)
    ear_movement = detect_ear_movement(results.face_landmarks)
    mouth_opening_speed = analyze_mouth_opening_speed(results.face_landmarks)
    thumb_movement = analyze_thumb_movement(results.left_hand_landmarks)
    waist_angle = analyze_waist_angle(results.pose_landmarks)
    finger_movement = analyze_finger_movement(results.left_hand_landmarks)
    lip_color = detect_lip_color(face_region)
    cheek_movement = analyze_cheek_movement(results.face_landmarks)
    full_body_movement = analyze_full_body_movement(results.pose_landmarks)
    hand_distance = analyze_hand_distance(results.pose_landmarks)
    
    # Hand analysis
    hand_count = 0
    hand_data = []
    if results.left_hand_landmarks or results.right_hand_landmarks:
        hand_count += (1 if results.left_hand_landmarks else 0) + (1 if results.right_hand_landmarks else 0)
        for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                finger_count = count_fingers(hand_landmarks)
                gesture = detect_gesture(hand_landmarks)
                
                gesture_counter[gesture] += 1
                for g in gesture_counter:
                    if g != gesture:
                        gesture_counter[g] = 0
                if gesture_counter[gesture] < 5:
                    gesture = "None"
                
                x = int(hand_landmarks.landmark[0].x * frame.shape[1])
                y = int(hand_landmarks.landmark[0].y * frame.shape[0])
                cv2.putText(frame, f'Fingers: {finger_count}', (x, y - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Gesture: {gesture}', (x, y - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                logging.info(f"Gesture detected: {gesture}, Fingers: {finger_count}")
                
                hand_data.append([(lm.x, lm.y) for lm in hand_landmarks.landmark])
    
    # Object detection
    objects_detected = []
    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = yolo_model.names[int(box.cls)]
            conf = box.conf.item()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            objects_detected.append(label)
            logging.info(f"Object detected: {label}, Confidence: {conf:.2f}")
    
    # Save data
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(), heart_rate, breathing_rate, body_speed, elbow_angle, facial_expression,
            gaze[0] if gaze else None, gaze[1] if gaze else None, blink, skin_color, tremor, head_movement,
            walking_pattern, pulse_wrist, sweating, diaphragm_movement, muscle_fatigue, body_balance,
            eye_movement, pupil_size, eyebrow_movement, face_symmetry, lip_movement, neck_angle,
            chest_movement, eye_distance, wrist_movement, leg_movement, facial_tics, ear_movement,
            mouth_opening_speed, thumb_movement, waist_angle, finger_movement, lip_color, cheek_movement,
            full_body_movement, hand_distance
        ])
    
    # Display metrics
    y_pos = 30
    metrics = [
        f'FPS: {int(fps)}', f'Hands: {hand_count}', f'Face Landmarks: {face_landmark_count}',
        f'Pose Landmarks: {pose_landmark_count}', f'Heart Rate: {int(heart_rate) if heart_rate else "N/A"} bpm',
        f'Breathing Rate: {int(breathing_rate) if breathing_rate else "N/A"} bpm', f'Elbow Angle: {int(elbow_angle)} deg',
        f'Facial Expression: {facial_expression}', f'Gaze: ({gaze[0]:.2f}, {gaze[1]:.2f})' if gaze else 'Gaze: N/A',
        f'Blink: {"Yes" if blink else "No"}', f'Skin Color: {skin_color:.2f}' if skin_color else 'Skin Color: N/A',
        f'Tremor: {tremor:.4f}', f'Head Movement: {head_movement:.4f}',
        f'Objects: {", ".join(objects_detected) if objects_detected else "None"}'
    ]
    for metric in metrics:
        cv2.putText(frame, metric, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 25
    cv2.putText(frame, 'q: Quit, s: Snapshot, r: Record', (10, frame.shape[0] - 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Record video
    if recording and out is not None:
        out.write(frame)
        cv2.putText(frame, 'Recording...', (frame.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imshow('All Features Detection', frame)
    
    # Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"snapshot_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Snapshot saved: {filename}")
        logging.info(f"Snapshot saved: {filename}")
    elif key == ord('r'):
        if recording:
            recording = False
            out.release()
            print("Recording stopped.")
            logging.info(f"Recording stopped.")
        else:
            recording = True
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_file = os.path.join(video_dir, f"video_{timestamp}.mp4")
            out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, 
                                 (int(cap.get(3)), int(cap.get(4))))
            print(f"Recording started: {video_file}")
            logging.info(f"Recording started: {video_file}")

# Release resources
if recording and out is not None:
    out.release()
cap.release()
cv2.destroyAllWindows()
holistic.close()
