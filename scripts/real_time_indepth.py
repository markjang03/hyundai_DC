"""
Module: real_time_indepth.py

Description:
This script performs an in-depth real-time driver confidence analysis using a webcam.
It analyzes emotions, calculates additional features like Eye Aspect Ratio (EAR), head tilt angle, and mouth opening, and visualizes results on a Streamlit-based dashboard.
The in-depth analysis includes detailed feature extraction and live confidence trend visualization.

Classes and Functions:
----------------------
1. calculate_EAR(landmarks, eye_indices)
    - Calculates the Eye Aspect Ratio (EAR) from facial landmarks.
    - Parameters:
        - landmarks: List of facial landmarks detected by Mediapipe.
        - eye_indices: List of indices corresponding to eye landmarks.
    - Returns:
        - float: EAR value.

2. draw_landmarks(frame, face_landmarks)
    - Draws facial landmarks and highlights key points on the given frame.
    - Parameters:
        - frame: OpenCV image frame.
        - face_landmarks: Mediapipe-detected facial landmarks.
    - Returns:
        - OpenCV image frame with drawn landmarks.

3. real_time_emotion_analysis_streamlit(video_placeholder, chart_placeholder)
    - Performs real-time emotion and feature analysis, displaying results on a Streamlit dashboard.
    - Parameters:
        - video_placeholder: Streamlit container for displaying video frames.
        - chart_placeholder: Streamlit container for displaying time-series charts.
    - Returns:
        - None

4. main()
    - Main function to initialize the Streamlit application.
    - Sets up the dashboard with start/stop buttons and placeholders for video and charts.

Features:
---------
1. Real-time emotion detection and confidence calculation.
2. Calculation of additional biometric features:
    - Eye Aspect Ratio (EAR)
    - Head tilt angle
    - Mouth opening level
3. Moving average smoothing for confidence trends.
4. Visualization of real-time video feed and confidence trends.
5. In-depth feature analysis and live updates on Streamlit.


Output:
-------
1. Real-time video feed with annotated emotions and extracted features.
2. Time-series chart showing driver confidence levels and biometric trends.
3. Advanced visualization of landmarks and real-time feature analysis.

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
import logging


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("real_time_indepth")

warnings.filterwarnings("ignore")

# Mediapipe reset
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def calculate_EAR(landmarks, eye_indices):

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

def draw_landmarks(frame, face_landmarks):


    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )


    for idx, landmark in enumerate(face_landmarks.landmark):
        if idx in [33, 160, 158, 133, 153, 144, 263, 387, 385, 362, 380, 373, 1, 234, 454, 13, 14]:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # 녹색 점

    return frame

def real_time_emotion_analysis_streamlit(video_placeholder, chart_placeholder):

    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(0)  # default webcam

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

    update_interval = 1 # meaning update the chart every second

    last_update_time = time.time()

    while st.session_state['real_time_running_indepth']:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to read from webcam.")
            break

        emotions = detector.detect_emotions(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_face = face_mesh.process(rgb_frame)

        for face in emotions:
            emotions_scores = face['emotions']
            dominant_emotion = max(emotions_scores, key=emotions_scores.get)
            confidence = emotions_scores[dominant_emotion]
            driver_confidence_level = emotion_confidence_mapping.get(dominant_emotion, 0.0)

            landmarks = results_face.multi_face_landmarks
            if landmarks:
                landmarks = landmarks[0].landmark
                left_EAR = calculate_EAR(landmarks, LEFT_EYE)
                right_EAR = calculate_EAR(landmarks, RIGHT_EYE)
                avg_EAR = (left_EAR + right_EAR) / 2.0
            else:
                avg_EAR = 0.0

            if landmarks:
                try:
                    nose_tip = landmarks[1]
                    left_ear = landmarks[234]
                    right_ear = landmarks[454]
                    delta_x = right_ear.x - left_ear.x
                    delta_y = right_ear.y - left_ear.y
                    head_tilt_angle = np.degrees(np.arctan2(delta_y, delta_x))
                except IndexError as e:
                    head_tilt_angle = 0.0
            else:
                head_tilt_angle = 0.0


            if landmarks:
                try:
                    top_lip = landmarks[13]
                    bottom_lip = landmarks[14]
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

            # show the numbers on the frame
            (x, y, w, h) = face['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{dominant_emotion} ({confidence:.2f})",
                        (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Confidence Level: {driver_confidence_level:.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2)

            # EAR, Head Tilt Angle, Mouth Opening 주석 추가
            cv2.putText(frame, f"Avg EAR: {avg_EAR:.2f}",
                        (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Head Tilt Angle: {head_tilt_angle:.2f}°",
                        (x, y + h + 45), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Mouth Opening: {mouth_opening:.2f}",
                        (x, y + h + 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

            if landmarks:
                frame = draw_landmarks(frame, results_face.multi_face_landmarks[0])

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

            fatigue_factor = (0.3 - avg_EAR) * 0.2
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
                if 'chart_data_full_indepth' not in st.session_state:
                    st.session_state['chart_data_full_indepth'] = new_data
                else:
                    st.session_state['chart_data_full_indepth'] = pd.concat([st.session_state['chart_data_full_indepth'], new_data], ignore_index=True)
                chart_placeholder.line_chart(st.session_state['chart_data_full_indepth'].set_index('time_seconds'))
                chart_data_list = []
                st.session_state['last_update_time_indepth'] = current_time

        # this is to avoid cpu overload (recommenede )
        time.sleep(0.03)

def main():
    """
    main func
    """
    st.title("real time drive confidence meausre (In-Depth)")

    video_placeholder = st.empty()
    chart_placeholder = st.empty()

    # start stop button
    if 'real_time_running_indepth' not in st.session_state:
        st.session_state['real_time_running_indepth'] = False

    if st.button("Start Real-Time Analysis (In-Depth)"):
        st.session_state['real_time_running_indepth'] = True
        st.session_state['chart_data_full_indepth'] = pd.DataFrame(columns=['time_seconds', 'driver_confidence_level'])
        st.session_state['last_update_time_indepth'] = time.time()
        real_time_emotion_analysis_streamlit(video_placeholder, chart_placeholder)

    if st.button("Stop Real-Time Analysis (In-Depth)"):
        st.session_state['real_time_running_indepth'] = False

if __name__ == "__main__":
    main()