# scripts/real_time_indepth.py

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

# 로깅 설정
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("real_time_indepth")

# 경고 메시지 무시 (선택 사항)
warnings.filterwarnings("ignore")

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Mediapipe 그리기 유틸리티 초기화
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def calculate_EAR(landmarks, eye_indices):
    """
    Eye Aspect Ratio (EAR)를 계산하는 함수입니다.

    :param landmarks: 얼굴 랜드마크
    :param eye_indices: 눈의 랜드마크 인덱스 리스트
    :return: EAR 값
    """
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
    """
    얼굴 랜드마크를 프레임에 그리는 함수입니다.

    :param frame: OpenCV 프레임
    :param face_landmarks: Mediapipe가 감지한 얼굴 랜드마크
    :return: 랜드마크가 그려진 프레임
    """
    # 랜드마크 그리기
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )

    # 주요 랜드마크에 점 추가 (눈, 코, 입 등)
    for idx, landmark in enumerate(face_landmarks.landmark):
        if idx in [33, 160, 158, 133, 153, 144, 263, 387, 385, 362, 380, 373, 1, 234, 454, 13, 14]:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # 녹색 점

    return frame

def real_time_emotion_analysis_streamlit(video_placeholder, chart_placeholder):
    """
    실시간 감정 분석을 수행하고, 결과를 Streamlit에 시각화하는 함수입니다.

    :param video_placeholder: Streamlit에서 비디오 프레임을 표시할 자리
    :param chart_placeholder: Streamlit에서 차트를 표시할 자리
    """
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(0)  # 기본 웹캠 사용

    if not cap.isOpened():
        st.error("Cannot open webcam.")
        return

    # 감정-자신감 매핑
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

    # 데이터 수집을 위한 리스트 초기화
    chart_data_list = []

    # Moving Average를 위한 창 크기 설정
    window_size = 5
    confidence_levels = []

    # 눈 EAR 계산을 위한 랜드마크 인덱스 (왼쪽, 오른쪽 눈)
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [263, 387, 385, 362, 380, 373]

    # 실시간 데이터 업데이트 주기 설정 (초 단위)
    update_interval = 1  # 1초마다 차트 업데이트

    last_update_time = time.time()

    while st.session_state['real_time_running_indepth']:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to read from webcam.")
            break

        # 얼굴 및 감정 분석
        emotions = detector.detect_emotions(frame)

        # 얼굴 랜드마크 분석
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_face = face_mesh.process(rgb_frame)

        for face in emotions:
            emotions_scores = face['emotions']
            dominant_emotion = max(emotions_scores, key=emotions_scores.get)
            confidence = emotions_scores[dominant_emotion]
            driver_confidence_level = emotion_confidence_mapping.get(dominant_emotion, 0.0)

            # 추가 피처 추출
            landmarks = results_face.multi_face_landmarks
            if landmarks:
                landmarks = landmarks[0].landmark
                # 왼쪽 눈 EAR
                left_EAR = calculate_EAR(landmarks, LEFT_EYE)
                # 오른쪽 눈 EAR
                right_EAR = calculate_EAR(landmarks, RIGHT_EYE)
                # 평균 EAR
                avg_EAR = (left_EAR + right_EAR) / 2.0
            else:
                avg_EAR = 0.0  # 얼굴 랜드마크를 찾지 못한 경우

            # 머리 기울기 계산 (간단한 예시: 얼굴 랜드마크를 이용한 기울기 추정)
            if landmarks:
                try:
                    nose_tip = landmarks[1]  # 1번 랜드마크: 코 끝
                    left_ear = landmarks[234]  # 왼쪽 귀
                    right_ear = landmarks[454]  # 오른쪽 귀
                    delta_x = right_ear.x - left_ear.x
                    delta_y = right_ear.y - left_ear.y
                    head_tilt_angle = np.degrees(np.arctan2(delta_y, delta_x))
                except IndexError as e:
                    head_tilt_angle = 0.0
            else:
                head_tilt_angle = 0.0

            # 입 열림 정도 계산 (간단한 예시: 입의 수직 길이와 수평 길이 비율)
            if landmarks:
                try:
                    top_lip = landmarks[13]  # 13번 랜드마크: 입 위쪽
                    bottom_lip = landmarks[14]  # 14번 랜드마크: 입 아래쪽
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

            # 프레임에 감정 및 신뢰도 표시
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

            # 랜드마크 시각화
            if landmarks:
                frame = draw_landmarks(frame, results_face.multi_face_landmarks[0])

            # 신뢰도 수준을 리스트에 추가하여 Moving Average 계산
            confidence_levels.append(driver_confidence_level)
            if len(confidence_levels) > window_size:
                confidence_levels.pop(0)
            smoothed_confidence = np.mean(confidence_levels)

            # 변화율 계산
            if len(confidence_levels) >= 2:
                change_rate = abs(confidence_levels[-1] - confidence_levels[-2])
            else:
                change_rate = 0.0

            # 최종 자신감 수준 계산
            final_confidence = smoothed_confidence - change_rate
            final_confidence = max(final_confidence, 0.0)

            # 피로도 반영 (EAR이 낮을수록 피로도 증가하여 자신감 감소)
            fatigue_factor = (0.3 - avg_EAR) * 0.2  # EAR이 0.3 이하일 경우 피로도 증가
            fatigue_factor = np.clip(fatigue_factor, -0.2, 0.2)
            final_confidence += fatigue_factor
            final_confidence = np.clip(final_confidence, 0.0, 1.0)

            # 최종 자신감 수준을 데이터 리스트에 추가
            chart_data_list.append({
                'time_seconds': round(timestamp, 2),
                'driver_confidence_level': final_confidence
            })

        # 프레임을 RGB로 변환하고 Streamlit에 표시
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")

        # 차트 업데이트 주기 체크
        current_time = time.time()
        if current_time - last_update_time >= update_interval:
            if chart_data_list:
                # 리스트를 데이터프레임으로 변환
                new_data = pd.DataFrame(chart_data_list)
                # 기존 차트 데이터에 추가
                if 'chart_data_full_indepth' not in st.session_state:
                    st.session_state['chart_data_full_indepth'] = new_data
                else:
                    st.session_state['chart_data_full_indepth'] = pd.concat([st.session_state['chart_data_full_indepth'], new_data], ignore_index=True)
                # 차트 업데이트
                chart_placeholder.line_chart(st.session_state['chart_data_full_indepth'].set_index('time_seconds'))
                # 리스트 초기화
                chart_data_list = []
                st.session_state['last_update_time_indepth'] = current_time

        # CPU 사용량 줄이기 위해 잠시 대기
        time.sleep(0.03)

def main():
    """
    Streamlit 애플리케이션의 메인 함수입니다.
    """
    st.title("실시간 운전자 자신감 측정기 (In-Depth)")

    # 비디오 및 차트 플레이스홀더 생성
    video_placeholder = st.empty()
    chart_placeholder = st.empty()

    # 시작/정지 버튼
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