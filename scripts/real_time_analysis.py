# scripts/real_time_analysis.py

import cv2
from fer import FER
import pandas as pd
import time
import os
import streamlit as st
import matplotlib.pyplot as plt
import warnings

# Suppress warnings (optional)
warnings.filterwarnings("ignore")

def real_time_emotion_analysis_streamlit(video_placeholder, chart_placeholder):
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(0)  # Use default webcam

    if not cap.isOpened():
        st.error("Cannot open webcam.")
        return

    emotion_confidence_mapping = {
        'happy': 1.0,
        'neutral': 0.5,
        'surprise': 0.3,
        'sad': 0.0,
        'angry': 0.2,
        'fear': 0.2,
        'disgust': 0.1,
        'contempt': 0.1
    }

    results = []
    start_time = time.time()

    # Initialize the chart
    chart_data = pd.DataFrame(columns=['time_seconds', 'driver_confidence_level'])

    # Streamlit's ability to handle live updates
    while st.session_state['real_time_running']:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to read from webcam.")
            break

        # Emotion analysis
        emotions = detector.detect_emotions(frame)
        for face in emotions:
            emotions_scores = face['emotions']
            dominant_emotion = max(emotions_scores, key=emotions_scores.get)
            confidence = emotions_scores[dominant_emotion]
            driver_confidence_level = emotion_confidence_mapping.get(dominant_emotion, 0.0)
            timestamp = time.time() - start_time
            results.append({
                'time_seconds': round(timestamp, 2),
                'emotion': dominant_emotion,
                'confidence': confidence,
                'driver_confidence_level': driver_confidence_level
            })

            # Display emotion and confidence level on frame
            (x, y, w, h) = face['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{dominant_emotion} ({confidence:.2f})",
                        (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Confidence Level: {driver_confidence_level:.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2)

        # Convert frame to RGB and display in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")

        # Update the chart
        if results:
            df = pd.DataFrame(results)
            chart_data = df[['time_seconds', 'driver_confidence_level']]
            chart_placeholder.line_chart(chart_data.set_index('time_seconds'))

        # Small sleep to reduce CPU usage
        time.sleep(0.03)

    cap.release()
    cv2.destroyAllWindows()

    # If "Stop" was clicked, save the chart into the 'outputs' folder
    if not st.session_state['real_time_running']:
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        chart_file_path = os.path.join(output_dir, f'real_time_chart_{int(time.time())}.png')

        # Plot the final chart with matplotlib
        plt.figure(figsize=(10, 6))
        plt.plot(chart_data['time_seconds'], chart_data['driver_confidence_level'], marker='o')
        plt.title('Driver Confidence Level Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Confidence Level')
        plt.grid(True)
        plt.savefig(chart_file_path)  # Save the chart as a PNG file
        plt.close()

        st.success(f"Real-time analysis stopped. Chart saved to {chart_file_path}.")
