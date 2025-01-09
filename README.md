# Hyundai Driver Confidence Measure

Welcome to the Hyundai Driver Confidence Measure, a comprehensive tool designed to analyze driver emotions and assess confidence levels in real-time and from pre-recorded video footage. This application leverages advanced computer vision and machine learning techniques to provide insightful analytics on driver behavior and state.

Table of Contents

	1.	Project Overview
	2.	Features
	3.	Technology Stack
	4.	Installation
	5.	Usage
	•	1. Video Analysis
	•	2. Real-Time Analysis
	•	3. Results & Report
	6.	Emotion Detection & Driver Confidence Calculation
	•	Emotion Detection
	•	Feature Extraction
	•	Driver Confidence Calculation
	7.	Data Flow
	8.	Troubleshooting
	9.	Contributing
	10.	License

Project Overview

The Hyundai Driver Confidence Measure is designed to monitor and analyze driver emotions and behaviors to assess confidence levels. By processing video inputs, either pre-recorded or in real-time, the application extracts relevant facial features and emotions, calculates driver confidence, and presents the results through intuitive visualizations and reports.

Features

	•	Video Analysis: Upload and analyze pre-recorded videos to detect emotions and extract facial features.
	•	Real-Time Analysis: Monitor and analyze driver emotions and confidence levels in real-time using a webcam.
	•	Comprehensive Reports: Generate detailed reports showcasing emotion trends, confidence levels, and other behavioral metrics.
	•	Visualization: Interactive charts and graphs to visualize emotion distributions and confidence trends over time.
	•	Annotation: Annotate video frames with detected emotions and confidence levels for better understanding.

Technology Stack

	•	Programming Language: Python
	•	Web Framework: Streamlit
	•	Computer Vision: OpenCV, Mediapipe
	•	Emotion Detection: FER (Facial Emotion Recognition)
	•	Data Processing: Pandas, NumPy
	•	Visualization: Matplotlib
	•	Logging: Python logging module

Installation

Prerequisites

	•	Python 3.7 or higher
	•	pip (Python package installer)

Steps

	1.	Clone the Repository:

git clone https://github.com/markjang03/hyundai-driver-confidence-measurer.git
cd hyundai-driver-confidence-measurer


	2.	Create a Virtual Environment:
It’s recommended to use a virtual environment to manage dependencies.

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


	3.	Install Dependencies:

pip install -r requirements.txt

If requirements.txt is not provided, install the necessary packages manually:

pip install streamlit opencv-python fer mediapipe pandas numpy matplotlib tqdm 
copy and paste the line above in terminal

	4.	Verify Installation:
Ensure that all packages are installed correctly.

pip list


Usage

Running the Streamlit Application

Navigate to the project directory and run the Streamlit app:

streamlit run gui/app.py -> this will only run locally

Application Tabs

The application is divided into three main tabs:

	1.	Video Analysis
	2.	Real-Time Analysis
	3.	Results & Report

1. Video Analysis

Purpose: Upload and analyze pre-recorded videos to detect emotions and extract facial features.

Steps:

	1.	Upload Video: Click on the “Upload video for analysis” button and select a video file (.mp4, .avi, .mov).
	2.	Start Analysis: After uploading, click the “Start Analysis” button to begin processing. The application will:
	•	Extract frames from the video at a specified FPS (Frames Per Second).
	•	Analyze each frame to detect emotions and extract facial features.
	•	Annotate frames with detected emotions and confidence levels.
	•	Compile the annotated frames back into a final video.
	•	Generate a comprehensive report.
	3.	View Results: Once analysis is complete, the final video and emotion trends will be displayed.

2. Real-Time Analysis (incomplete {this would run but wont save the result ima work on it soon})

Purpose: Monitor and analyze driver emotions and confidence levels in real-time using a webcam.

Steps:
	1.	Start Analysis: Click the “Start Real-Time Analysis” button to begin monitoring. 
 The app will:
	•	Capture video frames from your webcam.
	•	Detect emotions and extract facial features in real-time.
	•	Display the video feed with annotated emotions and confidence levels.
	•	Plot confidence trends over time.
	2.	Stop Analysis: Click the “Stop Real-Time Analysis” button to end monitoring. The application will save the confidence trends chart.

3. Results & Report

Purpose: View analyzed data, visualizations, and download comprehensive reports.

Features:
	•	Analyzed Emotion Data: View a dataframe containing emotion scores and extracted features for each frame.
	•	Emotion Trends Graph: Visual representation of driver confidence levels over time.
	•	Emotion Distribution Graph: Distribution of detected emotions throughout the video.
	•	Driver Confidence Level Over Time: Line chart showcasing confidence level variations.
	•	Individual Metrics Over Time: Separate charts for Emotion Confidence Score, Average EAR, Head Tilt Angle, and Mouth Opening.
	•	Final Video: Play the annotated final video.
	•	Download Report: Download a PDF report summarizing the analysis.

Emotion Detection & Driver Confidence Calculation

Emotion Detection

The application uses the FER (Facial Emotion Recognition) library in conjunction with Mediapipe to detect and analyze emotions from video frames.

	1.	Face Detection: FER uses mtcnn (Multi-task Cascaded Convolutional Networks) for robust face detection.
	2.	Emotion Analysis: For each detected face, FER predicts emotions such as happiness, sadness, anger, etc., along with confidence scores.

Feature Extraction

In addition to emotion detection, the application extracts several facial features to assess driver behavior:

	1.	Eye Aspect Ratio (EAR):
	•	Purpose: Measure eye openness to detect signs of fatigue.
	•	Calculation: Based on specific facial landmarks around the eyes, EAR is calculated using the distances between certain points.
	2.	Head Tilt Angle:
	•	Purpose: Determine the tilt of the driver’s head, which can indicate distraction or fatigue.
	•	Calculation: Calculated using the positions of the nose tip and ear landmarks to estimate the angle of head tilt.
	3.	Mouth Opening:
	•	Purpose: Assess mouth movements, which can indicate stress or fatigue.
	•	Calculation: Measured as the vertical distance between the top and bottom lip landmarks.

Driver Confidence Calculation

The driver confidence level is a composite score derived from detected emotions and extracted facial features. Here’s how it’s calculated:

	1.	Emotion Confidence Mapping:
	•	Each detected emotion is assigned a base confidence score based on predefined mappings.

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

	2.	Smoothed Confidence:
	•	A moving average (window size of 5) is applied to the confidence scores to smooth out fluctuations.
	3.	Change Rate:
	•	The absolute difference between consecutive smoothed confidence scores is calculated to detect rapid changes.
	4.	Final Confidence Level:
	•	Formula:
	•	Constraints: Clipped between 0.0 and 1.0 to maintain valid confidence levels.
	5.	Fatigue Factor:
	•	Purpose: Adjust confidence based on signs of fatigue (low EAR values).
	•	Calculation:
￼
	•	Threshold is set at 0.3. EAR values below this indicate potential fatigue.
	•	Scaling factor is 0.2, limiting the fatigue adjustment to ±0.2.
	•	Adjustment: The fatigue factor is added to the final confidence score and clipped between 0.0 and 1.0.
	6.	Final Confidence Score:
	•	Represents the overall confidence level of the driver, considering emotional state and signs of fatigue.

Summary of Calculation Steps

	1.	Emotion Detection: Identify dominant emotion and assign a base confidence score.
	2.	Feature Extraction: Calculate EAR, head tilt angle, and mouth opening.
	3.	Confidence Smoothing: Apply moving average to smooth confidence scores.
	4.	Change Rate Adjustment: Subtract change rate to account for rapid emotion shifts.
	5.	Fatigue Adjustment: Modify confidence based on EAR to account for fatigue.
	6.	Final Confidence: Obtain the final confidence score within the range [0.0, 1.0].

Data Flow

	1.	Video Input:
	•	Pre-Recorded Videos: Uploaded via the Streamlit interface.
	•	Real-Time Analysis: Captured directly from the webcam.
	2.	Frame Extraction:
	•	Frames are extracted from videos at a specified FPS using extract_frames.py.
	3.	Frame Analysis:
	•	Each frame is analyzed for emotions and facial features using analyze_frames.py.
	•	Results are saved to CSV files.
	4.	Annotation & Compilation:
	•	Frames are annotated with detected emotions and confidence levels using annotate_frames.py.
	•	Annotated frames are compiled back into a final video using compile_video.py.
	5.	Report Generation:
	•	Comprehensive reports are generated using generate_report.py, summarizing the analysis.
	6.	Visualization:
	•	Emotion trends and distributions are visualized using Matplotlib and displayed within the Streamlit app.

License

This project is licensed under the MIT License.
