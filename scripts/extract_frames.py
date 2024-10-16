# scripts/extract_frames.py

import cv2
import os
import sys
import argparse
from tqdm import tqdm

def extract_frames(video_path, output_folder, fps=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Attempting to open video file at: {video_path}")
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    video_fps = vidcap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {video_fps}")

    frame_interval = int(video_fps / fps) if fps > 0 else 1
    print(f"Frame interval: {frame_interval}")

    count = 0
    saved_count = 0

    with tqdm(total=total_frames, desc="Extracting Frames") as pbar:
        while True:
            success, image = vidcap.read()
            if not success:
                break
            if count % frame_interval == 0:
                frame_path = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
                cv2.imwrite(frame_path, image)
                saved_count += 1
            count += 1
            pbar.update(1)

    vidcap.release()
    print(f"Total {saved_count} frames extracted to {output_folder}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save extracted frames.')
    parser.add_argument('--fps', type=int, default=1, help='Frames per second to extract.')

    args = parser.parse_args()

    extract_frames(args.video_path, args.output_folder, args.fps)