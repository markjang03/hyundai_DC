"""
Module: extract_frames.py

Description:
This script extracts frames from a video file at a specified frame rate (fps) and saves them as image files in a designated folder.
It also generates a CSV file containing metadata for the extracted frames, such as frame number and corresponding filenames.


Classes and Functions:
----------------------
1. extract_frames(video_path, frames_folder, output_csv, fps=1)
    - Extracts frames from a video file at the specified frame rate.
    - Parameters:
        - video_path (str): Path to the input video file.
        - frames_folder (str): Path to the folder where frames will be saved.
        - output_csv (str): Path to the output CSV file containing frame metadata.
        - fps (int): Frames per second to extract (default: 1).
    - Returns:
        - None

2. parse_arguments()
    - Parses command-line arguments for script execution.
    - Arguments:
        - --video_path: Path to the input video file (required).
        - --frames_folder: Path to the folder to save frames (required).
        - --output_csv: Path to save the output CSV file with frame metadata (required).
        - --fps: Frames per second to extract (default: 1).
    - Returns:
        - argparse.Namespace: Parsed arguments.

3. main()
    - Entry point of the script. Parses arguments and invokes `extract_frames`.

Usage:
------
1. Prepare a video file for frame extraction.
2. Run this script to extract frames:
    $ python extract_frames.py --video_path <video_path> --frames_folder <frames_folder> --output_csv <csv_path> --fps <frame_rate>


Output:
-------
1. Extracted frames saved in the specified folder.
2. A CSV file with metadata for each extracted frame, including:
    - Frame number
    - Frame filename

Author: Mark jang
"""

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

    :param video_path: video file path
    :param frames_folder: path where the frames are saved
    :param output_csv: csv path
    :param fps
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
        video_fps = 1
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

    # CSV
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