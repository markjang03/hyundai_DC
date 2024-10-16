# scripts/analyze_frames.py

import os
import cv2
from fer import FER
import pandas as pd
from tqdm import tqdm
import joblib
import argparse

def analyze_frames(frames_folder, model_path, output_csv):
    detector = FER(mtcnn=True)
    results = []

    emotion_confidence_mapping = {
        'happy': 1.0,
        'neutral': 0.5,
        'surprise': 0.7,
        'sad': 0.3,
        'angry': 0.2,
        'fear': 0.2,
        'disgust': 0.1,
        'contempt': 0.1
    }

    frame_files = sorted(os.listdir(frames_folder))
    total_frames = len(frame_files)

    for frame_file in tqdm(frame_files, desc="Analyzing Frames"):
        frame_path = os.path.join(frames_folder, frame_file)
        img = cv2.imread(frame_path)
        if img is None:
            continue
        detected = detector.detect_emotions(img)
        for face in detected:
            emotions = face['emotions']
            dominant_emotion = max(emotions, key=emotions.get)
            confidence_score = emotions[dominant_emotion]
            driver_confidence_level = emotion_confidence_mapping.get(dominant_emotion, 0.0)
            results.append({
                'frame': frame_file,
                'emotion': dominant_emotion,
                'confidence': confidence_score,
                'driver_confidence_level': driver_confidence_level
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Analysis results saved to {output_csv}.")

    # Save model (if needed)
    joblib.dump(detector, model_path)
    print(f"FER model saved to {model_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze frames for emotions.")
    parser.add_argument('--frames_folder', type=str, required=True, help='Folder containing frames to analyze.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to save the model.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the analysis results.')
    args = parser.parse_args()
    analyze_frames(args.frames_folder, args.model_path, args.output_csv)
