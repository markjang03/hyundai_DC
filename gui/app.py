# gui/app.py

import streamlit as st
import os
import sys
import subprocess
import pandas as pd
import cv2
from PIL import Image
import time
from datetime import datetime

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

def extract_frames(video_path, output_folder, fps=1):
    cmd = [
        'python', 'scripts/extract_frames.py',
        '--video_path', video_path,
        '--output_folder', output_folder,
        '--fps', str(fps)
    ]
    subprocess.run(cmd, check=True)

def analyze_frames(frames_folder, model_path, output_csv):
    cmd = [
        'python', 'scripts/analyze_frames.py',
        '--frames_folder', frames_folder,
        '--model_path', model_path,
        '--output_csv', output_csv
    ]
    subprocess.run(cmd, check=True)

def annotate_frames(frames_folder, analysis_csv, output_folder):
    cmd = [
        'python', 'scripts/annotate_frames.py',
        '--frames_folder', frames_folder,
        '--analysis_csv', analysis_csv,
        '--output_folder', output_folder
    ]
    subprocess.run(cmd, check=True)

def compile_video(frames_folder, output_video, fps=1):
    cmd = [
        'python', 'scripts/compile_video.py',
        '--frames_folder', frames_folder,
        '--output_video', output_video,
        '--fps', str(fps)
    ]
    subprocess.run(cmd, check=True)

def visualize_results(analysis_csv, trends_image):
    cmd = [
        'python', 'scripts/visualize_results.py',
        '--analysis_csv', analysis_csv,
        '--trends_image', trends_image
    ]
    subprocess.run(cmd, check=True)

def generate_report_script(analysis_csv, trends_image, report_path):
    cmd = [
        'python', 'scripts/generate_report.py',
        '--analysis_csv', analysis_csv,
        '--trends_image', trends_image,
        '--report_path', report_path
    ]
    subprocess.run(cmd, check=True)

def main():
    st.set_page_config(page_title="Hyundai Driver Confidence Measurer", layout="wide")
    st.title("Hyundai Driver Confidence Measure")

    tabs = st.tabs(["Video Analysis", "Real-Time Analysis", "Results"])

    with tabs[0]:
        st.header("1. Video Analysis")
        uploaded_file = st.file_uploader("Upload video for analysis", type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            # Save the uploaded video file
            os.makedirs("data", exist_ok=True)
            video_path = "data/input_video.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("Video uploaded successfully!")

            if st.button("Start Analysis"):
                # Generate unique identifiers for outputs
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                extracted_frames_folder = f"data/extracted_frames_{timestamp}"
                model_path = f"models/fer_model_{timestamp}.pkl"
                analysis_csv = f"outputs/emotion_analysis_{timestamp}.csv"
                annotated_frames_folder = f"outputs/annotated_frames_{timestamp}"
                final_video = f"outputs/final_output_video_{timestamp}.mp4"
                trends_image = f"outputs/emotion_trends_{timestamp}.png"
                report_path = f"reports/emotion_analysis_report_{timestamp}.pdf"

                os.makedirs('models', exist_ok=True)
                os.makedirs('outputs', exist_ok=True)
                os.makedirs('reports', exist_ok=True)

                with st.spinner("Extracting frames..."):
                    extract_frames(video_path, extracted_frames_folder, fps=1)
                with st.spinner("Analyzing frames..."):
                    analyze_frames(extracted_frames_folder, model_path, analysis_csv)
                with st.spinner("Annotating frames..."):
                    annotate_frames(extracted_frames_folder, analysis_csv, annotated_frames_folder)
                with st.spinner("Compiling video..."):
                    compile_video(annotated_frames_folder, final_video, fps=1)
                with st.spinner("Visualizing results..."):
                    visualize_results(analysis_csv, trends_image)
                with st.spinner("Generating report..."):
                    generate_report_script(analysis_csv, trends_image, report_path)
                st.success("Video analysis complete!")

                st.video(final_video)

                st.image(trends_image, caption="Emotion Trends Over Time")

                with open(report_path, "rb") as f:
                    report = f.read()
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name=os.path.basename(report_path),
                        mime="application/pdf",
                        key=f"report_download_button_{timestamp}"  # Unique key
                    )

    with tabs[1]:
        st.header("2. Real-Time Analysis")

        # Placeholders for video feed and charts
        video_placeholder = st.empty()
        chart_placeholder = st.empty()

        if 'real_time_running' not in st.session_state:
            st.session_state['real_time_running'] = False

        def start_real_time_analysis():
            st.session_state['real_time_running'] = True


        def stop_real_time_analysis():
            st.session_state['real_time_running'] = False

        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state['real_time_running']:
                st.button("Start Real-Time Analysis", on_click=start_real_time_analysis)
        with col2:
            if st.session_state['real_time_running']:
                st.button("Stop Real-Time Analysis", on_click=stop_real_time_analysis)

        if st.session_state['real_time_running']:
            # Run the real-time analysis and display video feed
            from scripts.real_time_analysis import real_time_emotion_analysis_streamlit

            real_time_emotion_analysis_streamlit(video_placeholder, chart_placeholder)

    with tabs[2]:
        st.header("3. Results")
        st.subheader("Analyzed Emotion Data")

        # Find the latest analysis CSV file
        outputs = os.listdir('outputs')
        analysis_files = [f for f in outputs if f.startswith('emotion_analysis_') and f.endswith('.csv')]

        if analysis_files:
            latest_analysis = max(analysis_files, key=lambda x: os.path.getctime(os.path.join('outputs', x)))
            analysis_csv = os.path.join('outputs', latest_analysis)
            df = pd.read_csv(analysis_csv)

            # Add 'time_seconds' column if not present
            if 'time_seconds' not in df.columns:
                if 'frame' in df.columns:
                    df['frame_number'] = df['frame'].apply(lambda x: int(x.split('_')[1].split('.jpg')[0]))
                    df['time_seconds'] = df['frame_number']  # fps=1, so frame number equals seconds
                else:
                    df['time_seconds'] = df.index  # Fallback if no 'frame' column

            st.dataframe(df)

            # Emotion Trends Graph
            st.subheader("Emotion Trends Graph")
            trends_image = analysis_csv.replace('emotion_analysis_', 'emotion_trends_').replace('.csv', '.png')
            if os.path.exists(trends_image):
                st.image(trends_image)
            else:
                st.write("No trends image found.")

            # Driver Confidence Level Over Time
            st.subheader("Driver Confidence Level Over Time")
            confidence_over_time = df.groupby('time_seconds')['driver_confidence_level'].mean()
            st.line_chart(confidence_over_time)

            # Final Video
            st.subheader("Final Video")
            final_video = analysis_csv.replace('emotion_analysis_', 'final_output_video_').replace('.csv', '.mp4')
            if os.path.exists(final_video):
                st.video(final_video)
            else:
                st.write("No final video found.")


        else:
            st.write("No analysis results available.")


if __name__ == "__main__":
    main()
