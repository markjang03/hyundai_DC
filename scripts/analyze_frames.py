# scripts/analyze_frames.py

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

# 로깅 설정
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("analyze_frames")

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def calculate_EAR(landmarks, eye_indices):
    # Eye Aspect Ratio (EAR) 계산
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
    지정된 폴더의 프레임들을 분석하여 감정 및 추가 피처 데이터를 CSV 파일로 저장합니다.

    :param frames_folder: 프레임이 저장된 폴더 경로
    :param model_path: 사용하지 않음 (FER는 내부적으로 모델 관리)
    :param output_csv: 분석 결과를 저장할 CSV 파일 경로
    :param fps: 프레임 속도 (초당 프레임 수)
    """
    detector = FER(mtcnn=True)
    results = []

    # 감정-자신감 매핑 재정의
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

    # 눈 EAR 계산을 위한 랜드마크 인덱스 (왼쪽, 오른쪽 눈)
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [263, 387, 385, 362, 380, 373]

    # `extract_frames.py`가 생성한 CSV 파일을 읽어 프레임 목록 획득
    frames_csv = os.path.join(frames_folder, '..', 'emotion_analysis.csv')  # 예시 경로 수정 필요
    # 아니면 'extract_frames.py'에서 생성한 CSV 파일을 직접 전달받음
    # 현재 `analyze_frames.py`는 'emotion_analysis_{timestamp}.csv' 파일을 받음

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

        # 얼굴 감정 분석
        try:
            detected = detector.detect_emotions(img)
        except Exception as e:
            log.warning(f"Error detecting emotions in frame {frame_file}: {e}")
            detected = []

        # 얼굴 랜드마크 분석
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

            # 추가 피처 추출
            landmarks = results_face.multi_face_landmarks if results_face and results_face.multi_face_landmarks else None
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
                    log.warning(f"Error calculating head tilt angle in frame {frame_file}: {e}")
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
                    log.warning(f"Error calculating mouth opening in frame {frame_file}: {e}")
                    mouth_opening = 0.0
            else:
                mouth_opening = 0.0

            # time_seconds 계산 (프레임 번호 / fps)
            time_seconds = idx / fps

            results.append({
                'frame': frame_file,  # 'frame' 컬럼 추가
                'time_seconds': time_seconds,
                'emotion': dominant_emotion,
                'confidence': confidence_score,
                'driver_confidence_level': driver_confidence_level,
                'avg_EAR': avg_EAR,
                'head_tilt_angle': head_tilt_angle,
                'mouth_opening': mouth_opening
            })

    df = pd.DataFrame(results)
    # 항상 CSV 파일을 생성, 심지어 데이터가 없을 경우에도
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    log.info(f"Emotion and feature analysis results saved to {output_csv}")

    if df.empty:
        log.warning("No emotions detected in any frames.")
    else:
        # 감정 스무딩 (Moving Average)
        window_size = 5
        df['smoothed_confidence'] = df['driver_confidence_level'].rolling(window=window_size, min_periods=1).mean()

        # 변화율 계산
        df['confidence_change_rate'] = df['smoothed_confidence'].diff().abs().fillna(0)

        # 최종 자신감 수준 (스무딩된 자신감 - 변화율)
        df['final_confidence'] = df['smoothed_confidence'] - df['confidence_change_rate']
        df['final_confidence'] = df['final_confidence'].clip(lower=0.0, upper=1.0)

        # 피로도 반영 (EAR이 낮을수록 피로도가 증가하여 자신감을 감소시키는 요소)
        df['fatigue_factor'] = (0.3 - df['avg_EAR']) * 0.2  # EAR이 0.3 이하일 경우 피로도 증가
        df['fatigue_factor'] = df['fatigue_factor'].clip(lower=-0.2, upper=0.2)
        df['final_confidence'] += df['fatigue_factor']
        df['final_confidence'] = df['final_confidence'].clip(lower=0.0, upper=1.0)

        # 최종 자신감 수준을 다시 CSV 파일에 저장
        df.to_csv(output_csv, index=False)
        log.info(f"Final confidence levels updated in {output_csv}")

        # 감정 추세 그래프 저장
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