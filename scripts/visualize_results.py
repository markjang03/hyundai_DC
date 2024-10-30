# scripts/visualize_results.py

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
    분석된 데이터를 기반으로 추가적인 시각화 작업을 수행합니다.

    :param analysis_csv: 분석된 데이터가 저장된 CSV 파일 경로
    :param trends_image: 감정 추세 그래프 이미지 파일 경로
    """
    # 데이터 로드
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

    # 감정 분포 그래프 생성
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