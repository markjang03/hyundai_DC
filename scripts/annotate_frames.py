"""
Module: annotate_frames.py

Description:
This script processes video frames by annotating them with emotion analysis and biometric
features such as confidence level, average EAR, head tilt angle, and mouth opening.
The annotated frames are saved to an output folder for visualization.
It uses Mediapipe for facial landmark detection and the analysis results from a CSV file to overlay detailed information on each frame.

Classes and Functions:
----------------------
1. draw_landmarks(img, face_landmarks)
    - Draws facial landmarks and highlights key points on the given image.
    - Parameters:
        - img (numpy.ndarray): OpenCV image to annotate.
        - face_landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): Facial landmarks detected by Mediapipe.
    - Returns:
        - numpy.ndarray: Annotated image with facial landmarks.

2. annotate_frames(frames_folder, analysis_csv, output_folder)
    - Annotates video frames using emotion and biometric data from the analysis CSV file.
    - Parameters:
        - frames_folder (str): Path to the folder containing original video frames.
        - analysis_csv (str): Path to the CSV file containing emotion and feature analysis results.
        - output_folder (str): Path to save the annotated frames.
    - Returns:
        - None

3. parse_arguments()
    - Parses command-line arguments for script execution.
    - Arguments:
        - --frames_folder: Path to the folder containing original frames (required).
        - --analysis_csv: Path to the CSV file with emotion analysis data (required).
        - --output_folder: Path to save annotated frames (required).
    - Returns:
        - argparse.Namespace: Parsed arguments.

4. main()
    - Entry point of the script. Parses arguments and invokes `annotate_frames`.

Usage:
------
1. Analyze frames for emotion and feature data using a separate script.
2. Run this script to annotate the frames:
    $ python annotate_frames.py --frames_folder <frames_path> --analysis_csv <csv_path> --output_folder <output_path>

Output:
-------
1. Annotated frames saved in the specified output folder.
2. Each frame contains:
    - Emotion and confidence level.
    - Average EAR, head tilt angle, and mouth opening.
    - Visualized facial landmarks for added context.

Author: Mark Jang
"""
import cv2
import pandas as pd
import os
import argparse
from tqdm import tqdm
import numpy as np
import mediapipe as mp
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("annotate_frames")

# init Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

# init Mediapipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def draw_landmarks(img, face_landmarks):
    """
    Draws facial landmarks on the image.

    :param img: OpenCV image
    :param face_landmarks: Facial landmarks detected by Mediapipe
    :return: Image with landmarks drawn
    """
    # Draw the facial landmarks
    mp_drawing.draw_landmarks(
        image=img,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )

    # Highlight key landmarks (e.g., eyes, nose, mouth)
    for idx, landmark in enumerate(face_landmarks.landmark):
        if idx in [33, 160, 158, 133, 153, 144, 263, 387, 385, 362, 380, 373, 1, 234, 454, 13, 14]:
            x = int(landmark.x * img.shape[1])
            y = int(landmark.y * img.shape[0])
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)  # Green dots

    return img

def annotate_frames(frames_folder, analysis_csv, output_folder):
    """
    Annotates frames with driver's emotion and confidence level using the analyzed data.

    :param frames_folder: Path to the folder containing original frames
    :param analysis_csv: Path to the CSV file containing analyzed data
    :param output_folder: Path to save the annotated frames
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load analyzed data
    df = pd.read_csv(analysis_csv)

    # Sort frames based on the frame name
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

            # Add annotations to the frame
            text_emotion = f"Emotion: {emotion} | Confidence: {confidence:.2f}"
            text_EAR = f"Avg EAR: {avg_EAR:.2f}"
            text_head = f"Head Tilt Angle: {head_tilt_angle:.2f}°"
            text_mouth = f"Mouth Opening: {mouth_opening:.2f}"

            # Set text positions
            x, y = 10, 30
            cv2.putText(img, text_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, text_EAR, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, text_head, (x, y + 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, text_mouth, (x, y + 90), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # Perform facial landmark analysis
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results_face = face_mesh.process(rgb_img)

            if results_face.multi_face_landmarks:
                face_landmarks = results_face.multi_face_landmarks[0]
                img = draw_landmarks(img, face_landmarks)

        # Save the annotated frame
        output_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(output_path, img)

    print(f"Annotated frames saved to {output_folder}.")

def parse_arguments():
    """
    Parses command-line arguments.

    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Annotate frames with emotion and confidence data.")
    parser.add_argument("--frames_folder", type=str, required=True, help="Path to the folder containing original frames.")
    parser.add_argument("--analysis_csv", type=str, required=True, help="Path to the emotion analysis CSV file.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save annotated frames.")
    return parser.parse_args()

def main():
    """
    Main function to execute the frame annotation process.
    """
    args = parse_arguments()
    annotate_frames(args.frames_folder, args.analysis_csv, args.output_folder)

if __name__ == "__main__":
    main()