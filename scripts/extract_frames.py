# scripts/extract_frames.py

import cv2
import os
import argparse
import pandas as pd
from tqdm import tqdm
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("extract_frames")

def extract_frames(video_path, frames_folder, output_csv, fps=1):
    """
    비디오 파일에서 프레임을 추출하여 지정된 폴더에 저장하고, 프레임 정보를 CSV 파일로 저장합니다.

    :param video_path: 비디오 파일 경로
    :param frames_folder: 프레임을 저장할 폴더 경로
    :param output_csv: 프레임 정보를 저장할 CSV 파일 경로
    :param fps: 초당 추출할 프레임 수
    """
    if not os.path.exists(video_path):
        log.error(f"Video file does not exist: {video_path}")
        return

    os.makedirs(frames_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"Failed to open video file: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 1  # 기본값 설정
    frame_interval = int(video_fps / fps) if fps > 0 else 1

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted_frames = []

    log.info(f"Starting frame extraction from {video_path}")
    for idx in tqdm(range(frame_count), desc="Extracting Frames"):
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frame_filename = f"frame_{idx:06d}.jpg"
            frame_path = os.path.join(frames_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            extracted_frames.append({'frame_number': idx, 'frame_filename': frame_filename})

    cap.release()

    # CSV 파일로 저장
    df = pd.DataFrame(extracted_frames)
    df.to_csv(output_csv, index=False)
    log.info(f"Extracted {len(extracted_frames)} frames. Saved to {output_csv}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--frames_folder", type=str, required=True, help="Path to the folder where frames will be saved.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to the output CSV file containing frame information.")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to extract.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    extract_frames(args.video_path, args.frames_folder, args.output_csv, fps=args.fps)

if __name__ == "__main__":
    main()