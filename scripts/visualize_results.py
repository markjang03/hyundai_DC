"""
Module: visualize_results.py

Description:
This script generates visualizations and graphs from analyzed data.
It creates charts for emotion distribution and saves them as image files for further analysis or reporting.

Classes and Functions:
----------------------
1. visualize_results(analysis_csv, trends_image)
    - Performs additional visualization tasks based on the analyzed data.
    - Parameters:
        - analysis_csv (str): Path to the CSV file containing analyzed data.
        - trends_image (str): Path to the emotion trends graph image file.
    - Returns:
        - None

Features:
---------
1. Creates an emotion distribution bar graph based on the analysis data.
2. Saves the generated graph as an image for further use.
3. Handles exceptions such as missing files or empty data gracefully.

Output:
-------
1. A bar graph showing the distribution of emotions saved as an image.
2. The graph is saved in the same directory as the analysis CSV file.

Author: Mark Jang
"""
import argparse
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    PageBreak
)
from reportlab.lib.units import inch
import os
import matplotlib.pyplot as plt
import io
import sys

def visualize_results(analysis_csv, trends_image):
    """
    generate results

    :param analysis_csv:
    :param trends_image:
    """
    try:
        df = pd.read_csv(analysis_csv)
        if df.empty:
            print(f"No data found in {analysis_csv}. Skipping visualization.")
            return
    except FileNotFoundError:
        print(f"Analysis CSV file not found: {analysis_csv}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file {analysis_csv}: {e}")
        sys.exit(1)
    emotion_distribution = df['emotion'].value_counts()
    plt.figure(figsize=(8,6))
    emotion_distribution.plot(kind='bar', color='skyblue')
    plt.title('Emotion Distribution')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.tight_layout()
    distribution_image_path = analysis_csv.replace('emotion_analysis_', 'emotion_distribution_').replace('.csv', '.png')
    plt.savefig(distribution_image_path)
    plt.close()
    print(f"Emotion distribution graph saved to {distribution_image_path}")