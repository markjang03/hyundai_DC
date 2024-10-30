# scripts/compile_video.py

import cv2
import os
import argparse
from tqdm import tqdm

def compile_video(frames_folder, output_video, fps=1):
    """
    주석이 추가된 프레임을 합쳐 최종 비디오를 생성합니다.

    :param frames_folder: 주석이 추가된 프레임이 저장된 폴더 경로
    :param output_video: 생성될 비디오 파일 경로
    :param fps: 생성될 비디오의 FPS
    """
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.lower().endswith(('.jpg', '.png'))])

    if not frame_files:
        print(f"No frames found in {frames_folder}")
        return

    # 첫 번째 프레임을 읽어 비디오의 크기 결정
    first_frame_path = os.path.join(frames_folder, frame_files[0])
    frame = cv2.imread(first_frame_path)
    height, width, layers = frame.shape

    # 비디오 작성자 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for frame_file in tqdm(frame_files, desc="Compiling Video"):
        frame_path = os.path.join(frames_folder, frame_file)
        img = cv2.imread(frame_path)
        if img is None:
            print(f"Failed to read frame: {frame_path}")
            continue
        video.write(img)

    video.release()
    print(f"Final video saved to {output_video}.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compile annotated frames into a video.")
    parser.add_argument("--frames_folder", type=str, required=True, help="Path to the folder containing annotated frames.")
    parser.add_argument("--output_video", type=str, required=True, help="Path to save the compiled video.")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second for the output video.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    compile_video(args.frames_folder, args.output_video, fps=args.fps)

if __name__ == "__main__":
    main()