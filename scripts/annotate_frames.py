# scripts/annotate_frames.py

import os
import cv2
import pandas as pd
from tqdm import tqdm
import argparse

def annotate_frames(frames_folder, analysis_csv, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.read_csv(analysis_csv)

    frame_emotions = df.groupby('frame')[['emotion', 'confidence', 'driver_confidence_level']].apply(lambda x: x.to_dict(orient='records')).to_dict()

    for frame_file in tqdm(os.listdir(frames_folder), desc="Annotating Frames"):
        frame_path = os.path.join(frames_folder, frame_file)
        img = cv2.imread(frame_path)
        if img is None:
            continue

        emotions = frame_emotions.get(frame_file, [])
        y0, dy = 30, 30
        for i, emo in enumerate(emotions):
            text = f"{emo['emotion']} ({emo['confidence']:.2f})"
            confidence_text = f"Confidence Level: {emo['driver_confidence_level']:.2f}"
            y = y0 + i * dy * 2  # Adjusted spacing
            cv2.putText(img, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, confidence_text, (50, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        annotated_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(annotated_path, img)

    print(f"Annotated frames saved to {output_folder}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate frames with emotions.")
    parser.add_argument('--frames_folder', type=str, required=True, help='Folder containing frames to annotate.')
    parser.add_argument('--analysis_csv', type=str, required=True, help='CSV file with emotion analysis.')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save annotated frames.')
    args = parser.parse_args()
    annotate_frames(args.frames_folder, args.analysis_csv, args.output_folder)
