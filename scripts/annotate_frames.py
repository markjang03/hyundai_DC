# scripts/annotate_frames.py

import cv2
import pandas as pd
import os
import argparse
from tqdm import tqdm
import numpy as np
import mediapipe as mp
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("annotate_frames")

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

# Mediapipe 그리기 유틸리티 초기화
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def draw_landmarks(img, face_landmarks):
    """
    얼굴 랜드마크를 이미지에 그리는 함수입니다.

    :param img: OpenCV 이미지
    :param face_landmarks: Mediapipe가 감지한 얼굴 랜드마크
    :return: 랜드마크가 그려진 이미지
    """
    # 랜드마크 그리기
    mp_drawing.draw_landmarks(
        image=img,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )

    # 주요 랜드마크에 점 추가 (눈, 코, 입 등)
    for idx, landmark in enumerate(face_landmarks.landmark):
        if idx in [33, 160, 158, 133, 153, 144, 263, 387, 385, 362, 380, 373, 1, 234, 454, 13, 14]:
            x = int(landmark.x * img.shape[1])
            y = int(landmark.y * img.shape[0])
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)  # 녹색 점

    return img

def annotate_frames(frames_folder, analysis_csv, output_folder):
    """
    분석된 데이터를 기반으로 프레임에 운전자의 감정 및 자신감 수준을 주석으로 추가하여 저장합니다.

    :param frames_folder: 원본 프레임이 저장된 폴더 경로
    :param analysis_csv: 분석된 데이터가 저장된 CSV 파일 경로
    :param output_folder: 주석이 추가된 프레임을 저장할 폴더 경로
    """
    os.makedirs(output_folder, exist_ok=True)

    # 분석 데이터 로드
    df = pd.read_csv(analysis_csv)

    # 프레임 이름을 기준으로 정렬
    df_sorted = df.sort_values('frame').reset_index(drop=True)

    frame_files = sorted([f for f in os.listdir(frames_folder) if f.lower().endswith(('.jpg', '.png'))])

    for idx, frame_file in enumerate(tqdm(frame_files, desc="Annotating Frames")):
        frame_path = os.path.join(frames_folder, frame_file)
        img = cv2.imread(frame_path)
        if img is None:
            print(f"Failed to read frame: {frame_path}")
            continue

        if idx < len(df_sorted):
            row = df_sorted.iloc[idx]
            confidence = row['final_confidence']
            emotion = row['emotion']
            avg_EAR = row.get('avg_EAR', 0.0)
            head_tilt_angle = row.get('head_tilt_angle', 0.0)
            mouth_opening = row.get('mouth_opening', 0.0)

            # 주석 추가
            text_emotion = f"Emotion: {emotion} | Confidence: {confidence:.2f}"
            text_EAR = f"Avg EAR: {avg_EAR:.2f}"
            text_head = f"Head Tilt Angle: {head_tilt_angle:.2f}°"
            text_mouth = f"Mouth Opening: {mouth_opening:.2f}"

            # 텍스트 위치 설정
            x, y = 10, 30
            cv2.putText(img, text_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, text_EAR, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, text_head, (x, y + 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, text_mouth, (x, y + 90), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # 얼굴 랜드마크 분석
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results_face = face_mesh.process(rgb_img)

            if results_face.multi_face_landmarks:
                face_landmarks = results_face.multi_face_landmarks[0]
                img = draw_landmarks(img, face_landmarks)

        # 주석이 추가된 프레임 저장
        output_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(output_path, img)

    print(f"Annotated frames saved to {output_folder}.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Annotate frames with emotion and confidence data.")
    parser.add_argument("--frames_folder", type=str, required=True, help="Path to the folder containing original frames.")
    parser.add_argument("--analysis_csv", type=str, required=True, help="Path to the emotion analysis CSV file.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save annotated frames.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    annotate_frames(args.frames_folder, args.analysis_csv, args.output_folder)

if __name__ == "__main__":
    main()