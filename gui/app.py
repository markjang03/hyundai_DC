# gui/app.py

import streamlit as st
import os
import sys
import subprocess
import pandas as pd
import time
from datetime import datetime

# 상위 디렉토리를 sys.path에 추가하여 scripts 모듈을 인식하게 함
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def extract_frames(video_path, frames_folder, output_csv, fps=1):
    cmd = [
        'python', 'scripts/extract_frames.py',
        '--video_path', video_path,
        '--frames_folder', frames_folder,
        '--output_csv', output_csv,
        '--fps', str(fps)
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        st.write(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error(f"Error executing extract_frames.py:\n{e.stderr}")

def analyze_frames(frames_folder, model_path, output_csv, fps=1):
    cmd = [
        'python', 'scripts/analyze_frames.py',
        '--frames_folder', frames_folder,
        '--model_path', model_path,  # 현재 analyze_frames.py는 model_path를 사용하지 않음
        '--output_csv', output_csv,
        '--fps', str(fps)
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        st.write(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error(f"Error executing analyze_frames.py:\n{e.stderr}")

def annotate_frames(frames_folder, analysis_csv, output_folder):
    cmd = [
        'python', 'scripts/annotate_frames.py',
        '--frames_folder', frames_folder,
        '--analysis_csv', analysis_csv,
        '--output_folder', output_folder
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        st.write(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error(f"Error executing annotate_frames.py:\n{e.stderr}")

def compile_video(frames_folder, output_video, fps=1):
    cmd = [
        'python', 'scripts/compile_video.py',
        '--frames_folder', frames_folder,
        '--output_video', output_video,
        '--fps', str(fps)
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        st.write(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error(f"Error executing compile_video.py:\n{e.stderr}")

def generate_report_script(analysis_csv, trends_image, report_path):
    cmd = [
        'python', 'scripts/generate_report.py',
        '--analysis_csv', analysis_csv,
        '--trends_image', trends_image,
        '--report_path', report_path
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        st.write(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error(f"Error executing generate_report.py:\n{e.stderr}")

def main():
    st.set_page_config(page_title="Hyundai Driver Confidence Measurer", layout="wide")
    st.title("Hyundai Driver Confidence Measurer")

    # Python 실행 파일과 버전을 출력하여 올바른 환경에서 실행되고 있는지 확인
    st.write(f"**Python Executable:** {sys.executable}")
    st.write(f"**Python Version:** {sys.version}")

    tabs = st.tabs(["Video Analysis", "Real-Time Analysis", "Real-Time Analysis (In-Depth)", "Results & Report"])

    # Tab 1: Video Analysis
    with tabs[0]:
        st.header("1. Video Analysis")
        uploaded_file = st.file_uploader("Upload video for analysis", type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            # 업로드된 비디오 파일 저장
            os.makedirs("data", exist_ok=True)
            video_path = os.path.join("data", f"input_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("Video uploaded successfully!")

            if st.button("Start Analysis"):
                # 고유 식별자를 생성하여 출력 파일 경로 설정
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                frames_folder = f"data/extracted_frames_{timestamp}"
                analysis_csv = f"outputs/emotion_analysis_{timestamp}.csv"
                annotated_frames_folder = f"outputs/annotated_frames_{timestamp}"
                final_video = f"outputs/final_output_video_{timestamp}.mp4"
                trends_image = f"outputs/emotion_trends_{timestamp}.png"
                emotion_distribution_image = f"outputs/emotion_distribution_{timestamp}.png"
                report_path = f"reports/emotion_analysis_report_{timestamp}.pdf"

                os.makedirs('outputs', exist_ok=True)
                os.makedirs('reports', exist_ok=True)

                with st.spinner("Extracting frames..."):
                    extract_frames(video_path, frames_folder, analysis_csv, fps=1)  # fps 조절 가능

                with st.spinner("Analyzing frames..."):
                    # model_path는 현재 사용되지 않으므로 빈 문자열 전달
                    analyze_frames(frames_folder, model_path="", output_csv=analysis_csv, fps=1)

                with st.spinner("Annotating frames..."):
                    annotate_frames(frames_folder, analysis_csv, annotated_frames_folder)

                with st.spinner("Compiling video..."):
                    compile_video(annotated_frames_folder, final_video, fps=1)

                with st.spinner("Generating report..."):
                    generate_report_script(analysis_csv, trends_image, report_path)

                st.success("Video analysis complete!")

                # 최종 비디오 표시
                if os.path.exists(final_video):
                    st.video(final_video)
                else:
                    st.write("No final video found.")

                # 감정 추세 이미지 표시
                if os.path.exists(trends_image):
                    st.image(trends_image, caption="Emotion Trends Over Time")
                else:
                    st.write("No trends image found.")

    # Tab 2: Real-Time Analysis
    with tabs[1]:
        st.header("2. Real-Time Analysis")

        # 세션 상태 초기화
        if 'real_time_running' not in st.session_state:
            st.session_state['real_time_running'] = False

        # 비디오 피드와 차트를 위한 플레이스홀더 초기화
        video_placeholder = st.empty()
        chart_placeholder = st.empty()

        def start_real_time_analysis():
            st.session_state['real_time_running'] = True

        def stop_real_time_analysis():
            st.session_state['real_time_running'] = False

        # 시작 및 중지 버튼 배치
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state['real_time_running']:
                st.button("Start Real-Time Analysis", on_click=start_real_time_analysis)
        with col2:
            if st.session_state['real_time_running']:
                st.button("Stop Real-Time Analysis", on_click=stop_real_time_analysis)

        # 실시간 분석 실행
        if st.session_state['real_time_running']:
            from scripts.real_time_analysis import real_time_emotion_analysis_streamlit
            real_time_emotion_analysis_streamlit(video_placeholder, chart_placeholder)

    # Tab 3: Real-Time Analysis (In-Depth)
    with tabs[2]:
        st.header("3. Real-Time Analysis (In-Depth)")

        # 세션 상태 초기화
        if 'real_time_running_indepth' not in st.session_state:
            st.session_state['real_time_running_indepth'] = False

        # 비디오 피드와 차트를 위한 플레이스홀더 초기화
        video_placeholder_indepth = st.empty()
        chart_placeholder_indepth = st.empty()

        def start_real_time_analysis_indepth():
            st.session_state['real_time_running_indepth'] = True

        def stop_real_time_analysis_indepth():
            st.session_state['real_time_running_indepth'] = False

        # 시작 및 중지 버튼 배치
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state['real_time_running_indepth']:
                st.button("Start Real-Time Analysis (In-Depth)", on_click=start_real_time_analysis_indepth)
        with col2:
            if st.session_state['real_time_running_indepth']:
                st.button("Stop Real-Time Analysis (In-Depth)", on_click=stop_real_time_analysis_indepth)

        # 실시간 분석 실행
        if st.session_state['real_time_running_indepth']:
            from scripts.real_time_indepth import real_time_emotion_analysis_streamlit
            real_time_emotion_analysis_streamlit(video_placeholder_indepth, chart_placeholder_indepth)

    # Tab 4: Results & Report
    with tabs[3]:
        st.header("4. Results & Report")
        st.subheader("Analyzed Emotion Data")

        # 최신 분석 CSV 파일 찾기
        outputs = os.listdir('outputs')
        analysis_files = [f for f in outputs if f.startswith('emotion_analysis_') and f.endswith('.csv')]

        if analysis_files:
            latest_analysis = max(analysis_files, key=lambda x: os.path.getctime(os.path.join('outputs', x)))
            analysis_csv = os.path.join('outputs', latest_analysis)
            try:
                df = pd.read_csv(analysis_csv)
            except Exception as e:
                st.error(f"Error reading analysis CSV: {e}")
                df = pd.DataFrame()

            # 'time_seconds' 컬럼이 없는 경우 추가
            if 'time_seconds' not in df.columns:
                if 'frame' in df.columns:
                    df['frame_number'] = df['frame'].apply(lambda x: int(''.join(filter(str.isdigit, x))))
                    df['time_seconds'] = df['frame_number']  # fps=1, frame 번호가 초 단위
                else:
                    df['time_seconds'] = df.index  # 대체 방안

            st.dataframe(df)

            # 감정 추세 그래프 표시
            st.subheader("Emotion Trends Graph")
            trends_image = analysis_csv.replace('emotion_analysis_', 'emotion_trends_').replace('.csv', '.png')
            if os.path.exists(trends_image):
                st.image(trends_image, caption="Emotion Trends Over Time")
            else:
                st.write("No trends image found.")

            # 감정 분포 그래프 표시
            emotion_distribution_image = analysis_csv.replace('emotion_analysis_', 'emotion_distribution_').replace('.csv', '.png')
            if os.path.exists(emotion_distribution_image):
                st.image(emotion_distribution_image, caption="Emotion Distribution")
            else:
                st.write("No emotion distribution image found.")

            # 드라이버 신뢰도 수준 시간에 따른 표시
            st.subheader("Driver Confidence Level Over Time")
            if 'final_confidence' in df.columns:
                confidence_over_time = df.groupby('time_seconds')['final_confidence'].mean()
                st.line_chart(confidence_over_time)
            else:
                st.write("No final confidence data available.")

            # 개별 요소의 시계열 그래프 표시
            st.subheader("Individual Metrics Over Time")

            metrics = ['emotion', 'avg_EAR', 'head_tilt_angle', 'mouth_opening']
            metric_titles = {
                'emotion': 'Emotion Confidence Score',
                'avg_EAR': 'Average EAR',
                'head_tilt_angle': 'Head Tilt Angle',
                'mouth_opening': 'Mouth Opening'
            }

            for metric in metrics:
                st.write(f"### {metric_titles.get(metric, metric)}")
                if metric == 'emotion':
                    # 감정 점수를 숫자로 변환 (감정-자신감 매핑 사용)
                    emotion_confidence_mapping = {
                        'happy': 0.6,
                        'neutral': 0.5,
                        'surprise': 0.3,
                        'sad': 0.2,
                        'angry': 0.2,
                        'fear': 0.2,
                        'disgust': 0.1,
                        'contempt': 0.1
                    }
                    df['emotion_confidence_score'] = df['emotion'].map(emotion_confidence_mapping)
                    st.line_chart(df.set_index('time_seconds')['emotion_confidence_score'])
                else:
                    if metric in df.columns:
                        st.line_chart(df.set_index('time_seconds')[metric])
                    else:
                        st.write(f"No data available for {metric_titles.get(metric, metric)}.")

            # 최종 비디오 표시
            st.subheader("Final Video")
            final_video = analysis_csv.replace('emotion_analysis_', 'final_output_video_').replace('.csv', '.mp4')
            if os.path.exists(final_video):
                st.video(final_video)
            else:
                st.write("No final video found.")

            # 보고서 다운로드 버튼
            st.subheader("Download Report")
            report_path = analysis_csv.replace('emotion_analysis_', 'emotion_analysis_report_').replace('.csv', '.pdf')
            if os.path.exists(report_path):
                with open(report_path, "rb") as f:
                    report = f.read()
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name=os.path.basename(report_path),
                        mime="application/pdf",
                        key=f"report_download_button_{int(time.time())}"  # 고유 키
                    )
            else:
                st.write("No report available for download.")
        else:
            st.write("No analysis results available.")

if __name__ == "__main__":
    main()