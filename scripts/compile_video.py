"""
Module: compile_video.py

Description:
This script compiles a sequence of image frames from a specified folder into a video file.
The frames are read in sorted order, combined into a video, and saved at a user-specified path.
The script supports configuring the frames-per-second (fps) for the output video.


Classes and Functions:
----------------------
1. compile_video(frames_folder, output_video, fps=1)
    - Compiles image frames into a video file.
    - Parameters:
        - frames_folder (str): Path to the folder containing image frames.
        - output_video (str): Path to save the compiled video file.
        - fps (int): Frames per second for the output video (default: 1).
    - Returns:
        - None

2. parse_arguments()
    - Parses command-line arguments for script execution.
    - Arguments:
        - --frames_folder: Path to the folder containing image frames (required).
        - --output_video: Path to save the compiled video file (required).
        - --fps: Frames per second for the output video (default: 1).
    - Returns:
        - argparse.Namespace: Parsed arguments.

3. main()
    - Entry point of the script. Parses arguments and invokes `compile_video`.

Usage:
------
1. Prepare a folder of image frames in sequential order.
2. Run this script to compile the frames into a video:
    $ python compile_video.py --frames_folder <frames_path> --output_video <video_path> --fps <frame_rate>

Output:
-------
1. A video file saved at the specified output path.
2. The video contains all frames from the folder in sorted order, played at the specified fps.

Author: Mark Jang
"""
import cv2
import os
import argparse
from tqdm import tqdm

def compile_video(frames_folder, output_video, fps=1):
    """
    :param frames_folder: path for folder
    :param output_video: vid path
    :param fps
    """
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.lower().endswith(('.jpg', '.png'))])

    if not frame_files:
        print(f"No frames found in {frames_folder}")
        return

    # reads the first frame of the video to decide the size
    first_frame_path = os.path.join(frames_folder, frame_files[0])
    frame = cv2.imread(first_frame_path)
    height, width, layers = frame.shape

    # reset
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