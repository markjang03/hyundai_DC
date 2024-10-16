import os
import cv2
from tqdm import tqdm
import argparse

def compile_video(frames_folder, output_video, fps=1):
    frame_files = sorted(os.listdir(frames_folder))
    if not frame_files:
        print("No frames found to compile.")
        return

    first_frame_path = os.path.join(frames_folder, frame_files[0])
    frame = cv2.imread(first_frame_path)
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for frame_file in tqdm(frame_files, desc="Compiling Video"):
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        video.write(frame)

    video.release()
    print(f"Final video saved as {output_video}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile annotated frames into a video.")
    parser.add_argument('--frames_folder', type=str, required=True, help='Folder containing frames to compile.')
    parser.add_argument('--output_video', type=str, required=True, help='Path to save the compiled video.')
    parser.add_argument('--fps', type=int, default=1, help='Frames per second for the output video.')
    args = parser.parse_args()
    compile_video(args.frames_folder, args.output_video, args.fps)
