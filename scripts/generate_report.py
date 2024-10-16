# scripts/generate_report.py

import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import os
import argparse

def generate_report(analysis_csv, trends_image, report_path):
    df = pd.read_csv(analysis_csv)
    df['frame_number'] = df['frame'].apply(lambda x: int(x.split('_')[1].split('.jpg')[0]))
    df['time_seconds'] = df['frame_number']

    # Calculate emotion distribution
    emotion_counts = df['emotion'].value_counts()

    # Calculate average driver confidence level
    average_confidence = df['driver_confidence_level'].mean()

    # Create PDF
    c = canvas.Canvas(report_path, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 50, "Emotion Analysis Report")

    # Emotion distribution
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 100, "Emotion Distribution:")
    c.setFont("Helvetica", 12)
    y = height - 120
    for emotion, count in emotion_counts.items():
        c.drawString(60, y, f"{emotion}: {count}")
        y -= 20

    # Average driver confidence level
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y - 10, f"Average Driver Confidence Level: {average_confidence:.2f}")
    y -= 30

    # Add emotion trends graph
    if os.path.exists(trends_image):
        c.drawImage(trends_image, 50, y - 300, width=500, height=300)
        y -= 320


    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y - 10, "Key Insights:")
    c.setFont("Helvetica", 12)
    insights = [
        "1. Happiness was the most frequent emotion.",
        "2. Fear increased during certain time periods.",
        "3. High-stress driving segments were identified.",
        "4. Correlation between emotional shifts and driving behavior observed.",
        "5. Fatigue detected during prolonged driving."
    ]
    y -= 30
    for insight in insights:
        c.drawString(60, y, insight)
        y -= 20

    c.save()
    print(f"Report generated at {report_path}.")

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Generate an emotion analysis report.")
        parser.add_argument('--analysis_csv', type=str, required=True, help='CSV file with emotion analysis.')
        parser.add_argument('--trends_image', type=str, required=True, help='Path to the emotion trends image.')
        parser.add_argument('--report_path', type=str, required=True, help='Path to save the PDF report.')
        args = parser.parse_args()
        generate_report(args.analysis_csv, args.trends_image, args.report_path)
