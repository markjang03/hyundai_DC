"""
Module: analyze_frames.py
Description:
This script processes video frames for emotion and feature analysis,
specifically targeting driver confidence levels. It uses Mediapipe for facial landmark detection and FER for emotion recognition.
The results, including emotions, confidence levels, and additional biometric features, are saved to a CSV file.
The script also calculates smoothed confidence levels and visualizes trends over time in a graph.

Classes and Functions:
----------------------
1. calculate_EAR(landmarks, eye_indices)
    - Calculates the Eye Aspect Ratio (EAR) for given facial landmarks.
    - Parameters:
        - landmarks (list): List of facial landmarks detected by Mediapipe.
        - eye_indices (list): Indices of landmarks corresponding to an eye.
    - Returns:
        - float: EAR value. Returns 0.0 in case of an error.

2. analyze_frames(frames_folder, model_path, output_csv, fps=1)
    - Analyzes video frames for emotions and biometric features, then saves results to a CSV file.
    - Parameters:
        - frames_folder (str): Path to the folder containing video frames.
        - model_path (str): Path to the emotion detection model (currently unused).
        - output_csv (str): Path to save the analysis results in CSV format.
        - fps (int): Frame rate used to calculate timestamps for each frame.
    - Returns:
        - None

3. parse_arguments()
    - Parses command-line arguments for script execution.
    - Arguments:
        - --frames_folder: Path to the folder with extracted frames (required).
        - --model_path: Path to the emotion detection model (optional).
        - --output_csv: Path to save the analysis CSV file (required).
        - --fps: Frames per second for timestamp calculations (default: 1).
    - Returns:
        - argparse.Namespace: Parsed arguments.

4. main()
    - Entry point of the script. Parses arguments and invokes `analyze_frames`.

Usage:
------
1. Extract frames from a video using a separate script.
2. Run this script to analyze the frames:
    $ python analyze_frames.py --frames_folder <frames_path> --output_csv <output_csv_path> --fps <frame_rate>

Output:
-------
1. A CSV file containing:
    - Frame name
    - Time in seconds
    - Dominant emotion and confidence
    - Driver confidence level
    - Average EAR
    - Head tilt angle
    - Mouth opening
2. A visualization of driver confidence trends over time as a PNG image.

Author: Mark Jang
"""

import os
import cv2
from fer import FER
import pandas as pd
from tqdm import tqdm
import argparse
import logging
import sys
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("analyze_frames")

# Mediapipe reset
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def calculate_EAR(landmarks, eye_indices):
    # Eye Aspect Ratio (EAR) calculations
    try:
        A = np.linalg.norm(np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y]) -
                           np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y]))
        B = np.linalg.norm(np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y]) -
                           np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y]))
        C = np.linalg.norm(np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y]) -
                           np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y]))
        EAR = (A + B) / (2.0 * C)
    except IndexError as e:
        log.warning(f"Error calculating EAR: {e}")
        EAR = 0.0
    return EAR

def analyze_frames(frames_folder, model_path, output_csv, fps=1):
    """
    analyze the frames then save the data in csv
    :param frames_folder: path
    :param model_path: path
    :param output_csv: path
    :param fps: frame per second
    """
    detector = FER(mtcnn=True)
    results = []

    emotion_confidence_mapping = {
        'happy': 0.6,
        'neutral': 0.5,
        'surprise': 0.3,
        'sad': 0.2,
        'angry': 0.2,
        'fear': 0.2,
        'disgust': 0.1,
        'contempt': 0.1
    }

    # landmark index for EAR calculation
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [263, 387, 385, 362, 380, 373]

    frames_csv = os.path.join(frames_folder, '..', 'emotion_analysis_20241129_132101.csv') #this needs to be fixed

    # `extract_frames.py`에서 생성한 CSV 파일을 직접 사용
    frames_csv = output_csv.replace('emotion_analysis_', 'extracted_frames_').replace('.csv', '.csv')

    if not os.path.exists(frames_folder):
        log.error(f"Frames folder does not exist: {frames_folder}")
        sys.exit(1)

    # 프레임 파일 목록 불러오기
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.lower().endswith(('.jpg', '.png'))])

    if not frame_files:
        log.error(f"No image files found in {frames_folder}")
        sys.exit(1)

    log.info(f"Starting emotion and feature analysis on {len(frame_files)} frames.")

    for idx, frame_file in enumerate(tqdm(frame_files, desc="Analyzing Frames"), start=1):
        frame_path = os.path.join(frames_folder, frame_file)
        img = cv2.imread(frame_path)
        if img is None:
            log.warning(f"Failed to read frame: {frame_path}")
            continue

        # detect emotion here
        try:
            detected = detector.detect_emotions(img)
        except Exception as e:
            log.warning(f"Error detecting emotions in frame {frame_file}: {e}")
            detected = []

        # analyze landmarks
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            results_face = face_mesh.process(rgb_img)
        except Exception as e:
            log.warning(f"Error processing face landmarks in frame {frame_file}: {e}")
            results_face = None

        for face in detected:
            emotions = face.get('emotions', {})
            if not emotions:
                log.warning(f"No emotions detected in frame {frame_file}.")
                continue
            dominant_emotion = max(emotions, key=emotions.get)
            confidence_score = emotions[dominant_emotion]
            driver_confidence_level = emotion_confidence_mapping.get(dominant_emotion, 0.0)

            # additional features
            landmarks = results_face.multi_face_landmarks if results_face and results_face.multi_face_landmarks else None
            if landmarks:
                landmarks = landmarks[0].landmark

                left_EAR = calculate_EAR(landmarks, LEFT_EYE)

                right_EAR = calculate_EAR(landmarks, RIGHT_EYE)

                avg_EAR = (left_EAR + right_EAR) / 2.0
            else:
                avg_EAR = 0.0  # if no landmarks were found

            # head tilt calculation
            if landmarks:
                try:
                    nose_tip = landmarks[1]
                    left_ear = landmarks[234]
                    right_ear = landmarks[454]
                    delta_x = right_ear.x - left_ear.x
                    delta_y = right_ear.y - left_ear.y
                    head_tilt_angle = np.degrees(np.arctan2(delta_y, delta_x))
                except IndexError as e:
                    log.warning(f"Error calculating head tilt angle in frame {frame_file}: {e}")
                    head_tilt_angle = 0.0
            else:
                head_tilt_angle = 0.0

            # mouth opening frequencies
            if landmarks:
                try:
                    top_lip = landmarks[13]
                    bottom_lip = landmarks[14]
                    mouth_opening = np.linalg.norm([top_lip.x - bottom_lip.x, top_lip.y - bottom_lip.y])
                except IndexError as e:
                    log.warning(f"Error calculating mouth opening in frame {frame_file}: {e}")
                    mouth_opening = 0.0
            else:
                mouth_opening = 0.0

            # time_seconds calculation
            time_seconds = idx / fps

            results.append({
                'frame': frame_file,
                'time_seconds': time_seconds,
                'emotion': dominant_emotion,
                'confidence': confidence_score,
                'driver_confidence_level': driver_confidence_level,
                'avg_EAR': avg_EAR,
                'head_tilt_angle': head_tilt_angle,
                'mouth_opening': mouth_opening
            })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    log.info(f"Emotion and feature analysis results saved to {output_csv}")

    if df.empty:
        log.warning("No emotions detected in any frames.")
    else:
        # Moving Average
        window_size = 5
        df['smoothed_confidence'] = df['driver_confidence_level'].rolling(window=window_size, min_periods=1).mean()

        # calculate the diff
        df['confidence_change_rate'] = df['smoothed_confidence'].diff().abs().fillna(0)

        # final confidence calculation
        df['final_confidence'] = df['smoothed_confidence'] - df['confidence_change_rate']
        df['final_confidence'] = df['final_confidence'].clip(lower=0.0, upper=1.0)

        # small EAR meaning more fatigue which decreases confidence level
        df['fatigue_factor'] = (0.3 - df['avg_EAR']) * 0.2
        df['fatigue_factor'] = df['fatigue_factor'].clip(lower=-0.2, upper=0.2)
        df['final_confidence'] += df['fatigue_factor']
        df['final_confidence'] = df['final_confidence'].clip(lower=0.0, upper=1.0)
        df.to_csv(output_csv, index=False)
        log.info(f"Final confidence levels updated in {output_csv}")

        # plt
        trends_image_path = output_csv.replace('emotion_analysis_', 'emotion_trends_').replace('.csv', '.png')
        plt.figure(figsize=(12, 6))
        plt.plot(df['time_seconds'], df['final_confidence'], marker='o', label='Final Confidence Level')
        plt.title('Driver Confidence Level Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Confidence Level')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(trends_image_path)
        plt.close()
        log.info(f"Emotion trends graph saved to {trends_image_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze frames for emotions and features.")
    parser.add_argument("--frames_folder", type=str, required=True, help="Path to the folder containing extracted frames.")
    parser.add_argument("--model_path", type=str, required=False, default="", help="Path to the emotion detection model (unused).")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the analysis CSV file.")
    parser.add_argument("--fps", type=int, required=False, default=1, help="Frames per second used during frame extraction.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    analyze_frames(args.frames_folder, args.model_path, args.output_csv, fps=args.fps)

if __name__ == "__main__":
    main()