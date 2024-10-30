# scripts/generate_report.py

import argparse
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
)
from reportlab.lib.units import inch
import os
import matplotlib.pyplot as plt
import io
import sys

def generate_report(analysis_csv, trends_image, report_path):
    """
    Generates a PDF report including analyzed data and visualized graphs.

    :param analysis_csv: Path to the emotion analysis CSV file.
    :param trends_image: Path to the emotion trends image file.
    :param report_path: Path to save the generated PDF report.
    """
    # Load data
    try:
        df = pd.read_csv(analysis_csv)
        if df.empty:
            print(f"No data found in {analysis_csv}. Skipping report generation.")
            return
    except FileNotFoundError:
        print(f"Analysis CSV file not found: {analysis_csv}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file {analysis_csv}: {e}")
        sys.exit(1)

    # Set up ReportLab styles
    styles = getSampleStyleSheet()

    # Rename the custom Heading3 to avoid conflict
    styles.add(ParagraphStyle(name='CustomHeading3', parent=styles['Heading2'], fontSize=14, spaceAfter=10))

    styles.add(ParagraphStyle(name='CenterTitle', alignment=1, fontSize=18, spaceAfter=20))
    styles.add(ParagraphStyle(name='Justify', alignment=4, fontSize=12, spaceAfter=12))

    elements = []

    # Title
    title = Paragraph("Driver Confidence Analysis Report", styles['CenterTitle'])
    elements.append(title)

    # Report generation date
    from datetime import datetime
    report_date = Paragraph(f"Report Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
    elements.append(report_date)
    elements.append(Spacer(1, 12))

    # Section 1: Emotion Analysis Summary
    section1 = Paragraph("1. Emotion Analysis Summary", styles['Heading2'])
    elements.append(section1)

    # Emotion distribution table
    emotion_counts = df['emotion'].value_counts().reset_index()
    emotion_counts.columns = ['Emotion', 'Frequency']
    data = [emotion_counts.columns.tolist()] + emotion_counts.values.tolist()

    table = Table(data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (1,1), (-1,-1), 'CENTER'),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 12))

    # Emotion distribution graph insertion
    emotion_distribution_image = analysis_csv.replace('emotion_analysis_', 'emotion_distribution_').replace('.csv', '.png')
    if os.path.exists(emotion_distribution_image):
        im = RLImage(emotion_distribution_image, width=6*inch, height=4*inch)
        elements.append(im)
    else:
        elements.append(Paragraph("Emotion distribution graph image does not exist.", styles['Normal']))

    elements.append(Spacer(1, 12))

    # Section 2: Additional Features Analysis
    section2 = Paragraph("2. Additional Features Analysis", styles['Heading2'])
    elements.append(section2)

    # Additional features statistics table
    feature_stats = df[['avg_EAR', 'head_tilt_angle', 'mouth_opening']].describe().transpose().reset_index()
    feature_stats.columns = ['Feature', 'Count', 'Mean', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max']

    data = [feature_stats.columns.tolist()] + feature_stats.values.tolist()

    table = Table(data, hAlign='LEFT', colWidths=[100, 50, 50, 50, 50, 50, 50, 50])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgreen),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (1,1), (-1,-1), 'CENTER'),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 12))

    # Section 3: Time Series Analysis
    section3 = Paragraph("3. Time Series Analysis", styles['Heading2'])
    elements.append(section3)

    # Emotion trends graph insertion
    if os.path.exists(trends_image):
        im = RLImage(trends_image, width=6*inch, height=3*inch)
        elements.append(im)
    else:
        elements.append(Paragraph("Emotion trends graph image does not exist.", styles['Normal']))

    elements.append(Spacer(1, 12))

    # Time series data summary table
    time_series_stats = df[['final_confidence']].describe().reset_index()
    time_series_stats.columns = ['Statistic', 'Final Confidence']

    data = [time_series_stats.columns.tolist()] + time_series_stats.values.tolist()

    table = Table(data, hAlign='LEFT', colWidths=[100, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.pink),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (1,1), (-1,-1), 'CENTER'),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 12))

    # Section 4: Time Series Analysis of Individual Elements
    section4 = Paragraph("4. Time Series Analysis of Individual Elements", styles['Heading2'])
    elements.append(section4)

    # Metrics and their titles
    metrics = ['emotion', 'avg_EAR', 'head_tilt_angle', 'mouth_opening']
    metric_titles = {
        'emotion': 'Emotion Score',
        'avg_EAR': 'Average EAR',
        'head_tilt_angle': 'Head Tilt Angle',
        'mouth_opening': 'Mouth Opening Level'
    }

    # Emotion to confidence mapping
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

    for metric in metrics:
        section = Paragraph(f"4.{metrics.index(metric)+1} {metric_titles.get(metric, metric)}", styles['CustomHeading3'])
        elements.append(section)

        plt.figure(figsize=(12, 4))
        if metric == 'emotion':
            # Convert emotion scores to numerical values using the mapping
            df['emotion_confidence_score'] = df['emotion'].map(emotion_confidence_mapping)
            plt.plot(df['time_seconds'], df['emotion_confidence_score'], marker='o', label='Emotion Confidence Score')
            plt.ylabel('Confidence Score')
        else:
            plt.plot(df['time_seconds'], df[metric], marker='o', label=metric_titles.get(metric, metric))
            plt.ylabel(metric_titles.get(metric, metric))

        plt.title(f"{metric_titles.get(metric, metric)} Over Time")
        plt.xlabel('Time (seconds)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot to a buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='PNG', bbox_inches='tight', pad_inches=0.1)
        plt.close()
        img_buffer.seek(0)
        im = RLImage(img_buffer, width=6*inch, height=3*inch)
        elements.append(im)
        elements.append(Spacer(1, 12))

    # Section 5: Final Confidence Score Calculation Formula
    section5 = Paragraph("5. Final Confidence Score Calculation Formula", styles['Heading2'])
    elements.append(section5)

    # Use multiple Paragraphs instead of nested <para> tags
    elements.append(Paragraph("The final confidence score is calculated using the following formula:", styles['Justify']))
    elements.append(Paragraph("<b>Final Confidence</b> = (Smoothed Confidence) - (Confidence Change Rate) + (Fatigue Factor)", styles['Normal']))
    elements.append(Paragraph(
        "Where,<br/>"
        "- <b>Smoothed Confidence</b>: The driver's confidence level smoothed over time (Moving Average)<br/>"
        "- <b>Confidence Change Rate</b>: The rate of change in confidence level (absolute difference from the previous value)<br/>"
        "- <b>Fatigue Factor</b>: Confidence reduction factor due to fatigue (decreases as average EAR lowers)<br/>"
        "The final confidence score is clipped between 0.0 and 1.0.",
        styles['Justify']
    ))
    elements.append(Spacer(1, 12))

    # Section 6: Conclusion and Recommendations
    section6 = Paragraph("6. Conclusion and Recommendations", styles['Heading2'])
    elements.append(section6)

    conclusion = Paragraph(
        "Through this analysis, we have comprehensively measured the driver's confidence by considering their emotional state and additional features. "
        "Integrating multiple features and performing time series analysis have enabled a more accurate and reliable confidence assessment. "
        "Future research should incorporate a wider range of features and enhance real-time analysis performance to further improve driver safety and convenience.",
        styles['Justify']
    )
    elements.append(conclusion)
    elements.append(Spacer(1, 12))

    # Section 7: Original Data Table
    section7 = Paragraph("7. Original Data Table", styles['Heading2'])
    elements.append(section7)

    # Insert the original data table (only top 100 rows for simplicity)
    max_rows = 100
    data = [df.columns.tolist()] + df.head(max_rows).values.tolist()

    table = Table(data, repeatRows=1, hAlign='LEFT', colWidths=[80]*len(df.columns))
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.gray),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 12))

    if len(df) > max_rows:
        elements.append(Paragraph(f"The original data exceeds {max_rows} rows; only the top {max_rows} rows are displayed.", styles['Normal']))

    # Build the PDF
    try:
        doc = SimpleDocTemplate(
            report_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )
        doc.build(elements)
        print(f"Report successfully saved to {report_path}.")
    except Exception as e:
        print(f"Error generating report: {e}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a PDF report from emotion analysis data.")
    parser.add_argument("--analysis_csv", type=str, required=True, help="Path to the emotion analysis CSV file.")
    parser.add_argument("--trends_image", type=str, required=True, help="Path to the emotion trends image file.")
    parser.add_argument("--report_path", type=str, required=True, help="Path to save the generated PDF report.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    generate_report(args.analysis_csv, args.trends_image, args.report_path)

if __name__ == "__main__":
    main()