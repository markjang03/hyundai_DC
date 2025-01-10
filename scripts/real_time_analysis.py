"""
Module: real_time_analysis.py

Description:
This script implements a real-time driver confidence analysis system using a webcam.
It analyzes emotions and calculates additional features such as EAR (Eye Aspect Ratio),
head tilt angle, and mouth opening. The results are visualized on a Streamlit-based dashboard in real time.
It uses Mediapipe for facial landmark detection and FER for emotion recognition.


Classes and Functions:
----------------------
1. calculate_EAR(landmarks, eye_indices)
    - Calculates the Eye Aspect Ratio (EAR) from facial landmarks.
    - Parameters:
        - landmarks: List of facial landmarks detected by Mediapipe.
        - eye_indices: List of indices corresponding to eye landmarks.
    - Returns:
        - float: EAR value.

2. real_time_emotion_analysis_streamlit(video_placeholder, chart_placeholder)
    - Performs real-time emotion analysis and displays results in a Streamlit app.
    - Parameters:
        - video_placeholder: Streamlit container for displaying video frames.
        - chart_placeholder: Streamlit container for displaying time-series charts.
    - Returns:
        - None

3. main()
    - Main function to initialize the Streamlit application.
    - Sets up the dashboard with start/stop buttons and placeholders for video and charts.

Features:
---------
1. Real-time emotion detection and driver confidence calculation.
2. Visualization of confidence trends over time.
3. Calculation of additional biometric features:
    - Eye Aspect Ratio (EAR)
    - Head tilt angle
    - Mouth opening level
4. Automatic saving of time-series data and confidence trend charts.


Output:
-------
1. Real-time video feed with annotated emotions and features.
2. Time-series chart displaying driver confidence levels over time.

Author: Mark Jang
"""
import cv2
from fer import FER
import pandas as pd
import time
import os
import streamlit as st
import matplotlib.pyplot as plt
import warnings
import numpy as np
import mediapipe as mp

warnings.filterwarnings("ignore")

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

def calculate_EAR(landmarks, eye_indices):
    """
    Calculates the Eye Aspect Ratio (EAR).

    :param landmarks: Facial landmarks
    :param eye_indices: List of indices for eye landmarks
    :return: EAR value
    """
    A = np.linalg.norm(np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y]) -
                       np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y]))
    B = np.linalg.norm(np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y]) -
                       np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y]))
    C = np.linalg.norm(np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y]) -
                       np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y]))
    EAR = (A + B) / (2.0 * C)
    return EAR

def real_time_emotion_analysis_streamlit(video_placeholder, chart_placeholder):
    """
    Performs real-time emotion analysis and visualizes the results on Streamlit.

    :param video_placeholder: Placeholder to display video frames in Streamlit
    :param chart_placeholder: Placeholder to display charts in Streamlit
    """
    if 'real_time_running' not in st.session_state:
        st.session_state['real_time_running'] = False

    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(0)  # Use default webcam

    if not cap.isOpened():
        st.error("Cannot open webcam.")
        return

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

    results = []
    start_time = time.time()
    chart_data_list = []
    window_size = 5
    confidence_levels = []


    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [263, 387, 385, 362, 380, 373]

    update_interval = 1  # Update chart every second

    last_update_time = time.time()

    while st.session_state['real_time_running']:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to read from webcam.")
            break

        emotions = detector.detect_emotions(frame)

        # Analyze facial landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_face = face_mesh.process(rgb_frame)

        for face in emotions:
            emotions_scores = face['emotions']
            dominant_emotion = max(emotions_scores, key=emotions_scores.get)
            confidence = emotions_scores[dominant_emotion]
            driver_confidence_level = emotion_confidence_mapping.get(dominant_emotion, 0.0)

            # Extract additional features
            landmarks = results_face.multi_face_landmarks
            if landmarks:
                landmarks = landmarks[0].landmark
                # Left eye EAR
                left_EAR = calculate_EAR(landmarks, LEFT_EYE)
                # Right eye EAR
                right_EAR = calculate_EAR(landmarks, RIGHT_EYE)
                # Average EAR
                avg_EAR = (left_EAR + right_EAR) / 2.0
            else:
                avg_EAR = 0.0  # No facial landmarks found

            if landmarks:
                try:
                    nose_tip = landmarks[1]  # Landmark 1: Nose tip
                    left_ear = landmarks[234]  # Left ear
                    right_ear = landmarks[454]  # Right ear
                    delta_x = right_ear.x - left_ear.x
                    delta_y = right_ear.y - left_ear.y
                    head_tilt_angle = np.degrees(np.arctan2(delta_y, delta_x))
                except IndexError as e:
                    head_tilt_angle = 0.0
            else:
                head_tilt_angle = 0.0

            if landmarks:
                try:
                    top_lip = landmarks[13]  # Landmark 13: Upper lip
                    bottom_lip = landmarks[14]  # Landmark 14: Lower lip
                    mouth_opening = np.linalg.norm([top_lip.x - bottom_lip.x, top_lip.y - bottom_lip.y])
                except IndexError as e:
                    mouth_opening = 0.0
            else:
                mouth_opening = 0.0

            timestamp = time.time() - start_time
            results.append({
                'time_seconds': round(timestamp, 2),
                'emotion': dominant_emotion,
                'confidence': confidence,
                'driver_confidence_level': driver_confidence_level,
                'avg_EAR': avg_EAR,
                'head_tilt_angle': head_tilt_angle,
                'mouth_opening': mouth_opening
            })

            (x, y, w, h) = face['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{dominant_emotion} ({confidence:.2f})",
                        (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Confidence Level: {driver_confidence_level:.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2)

            cv2.putText(frame, f"Avg EAR: {avg_EAR:.2f}",
                        (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Head Tilt Angle: {head_tilt_angle:.2f}°",
                        (x, y + h + 45), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Mouth Opening: {mouth_opening:.2f}",
                        (x, y + h + 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

            confidence_levels.append(driver_confidence_level)
            if len(confidence_levels) > window_size:
                confidence_levels.pop(0)
            smoothed_confidence = np.mean(confidence_levels)

            if len(confidence_levels) >= 2:
                change_rate = abs(confidence_levels[-1] - confidence_levels[-2])
            else:
                change_rate = 0.0

            final_confidence = smoothed_confidence - change_rate
            final_confidence = max(final_confidence, 0.0)

            fatigue_factor = (0.3 - avg_EAR) * 0.2  # Increase fatigue when EAR is below 0.3
            fatigue_factor = np.clip(fatigue_factor, -0.2, 0.2)
            final_confidence += fatigue_factor
            final_confidence = np.clip(final_confidence, 0.0, 1.0)

            chart_data_list.append({
                'time_seconds': round(timestamp, 2),
                'driver_confidence_level': final_confidence
            })

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")

        current_time = time.time()
        if current_time - last_update_time >= update_interval:
            if chart_data_list:
                new_data = pd.DataFrame(chart_data_list)
                if 'chart_data_full' not in st.session_state:

                    st.session_state['chart_data_full'] = new_data
                else:
                    st.session_state['chart_data_full'] = pd.concat([st.session_state['chart_data_full'], new_data], ignore_index=True)
                chart_placeholder.line_chart(st.session_state['chart_data_full'].set_index('time_seconds'))
                chart_data_list = []
                st.session_state['last_update_time'] = current_time

        # Reduce CPU usage
        time.sleep(0.03)

    cap.release()
    cv2.destroyAllWindows()

    if not st.session_state['real_time_running']:
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())
        chart_file_path = os.path.join(output_dir, f'real_time_chart_{timestamp}.png')

        if 'chart_data_full' in st.session_state and not st.session_state['chart_data_full'].empty:
            plt.figure(figsize=(12, 6))
            plt.plot(st.session_state['chart_data_full']['time_seconds'], st.session_state['chart_data_full']['driver_confidence_level'], marker='o', label='Final Confidence Level')
            plt.title('Driver Confidence Level Over Time')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Confidence Level')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(chart_file_path)
            plt.close()

            st.success(f"Real-time analysis stopped. Chart saved to {chart_file_path}.")
        else:
            st.warning("No chart data to save.")

def main():
    """
    Main function to initialize the Streamlit application.
    """
    st.title("Real-Time Driver Confidence Measurement")

    video_placeholder = st.empty()
    chart_placeholder = st.empty()

    if 'real_time_running' not in st.session_state:
        st.session_state['real_time_running'] = False

    if st.button("Start Real-Time Analysis"):
        st.session_state['real_time_running'] = True
        st.session_state['chart_data_full'] = pd.DataFrame(columns=['time_seconds', 'driver_confidence_level'])
        st.session_state['last_update_time'] = time.time()
        real_time_emotion_analysis_streamlit(video_placeholder, chart_placeholder)

    if st.button("Stop Real-Time Analysis"):
        st.session_state['real_time_running'] = False

if __name__ == "__main__":
    main()