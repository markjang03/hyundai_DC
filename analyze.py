"""
Module: analyze.py

NOT IN USE

Author: Mark Jang
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data into a DataFrame
file_path = 'outputs/emotion_analysis_20241129_160707.csv'  # Replace with your file path
df = pd.read_csv(file_path)
title = "Mark"
# Descriptive statistics
print(df.describe())

# Convert columns to numeric where necessary
numeric_columns = [
    "confidence", "driver_confidence_level", "avg_EAR", "head_tilt_angle",
    "mouth_opening", "smoothed_confidence", "confidence_change_rate",
    "final_confidence", "fatigue_factor"
]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Plotting: Driver confidence level over time
plt.figure(figsize=(12, 6))
sns.lineplot(x=df["time_seconds"], y=df["driver_confidence_level"], label="Driver Confidence Level")
plt.title(f"{title} - Driver Confidence Level Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Driver Confidence Level")
plt.legend()
plt.grid(True)
plt.show()

# Plotting: Average EAR (Eye Aspect Ratio) over time
plt.figure(figsize=(12, 6))
sns.lineplot(x=df["time_seconds"], y=df["avg_EAR"], label="Average EAR")
plt.title(f"{title} - Average EAR Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Average EAR")
plt.legend()
plt.grid(True)
plt.show()

# Plotting: Head tilt angle over time
plt.figure(figsize=(12, 6))
sns.lineplot(x=df["time_seconds"], y=df["head_tilt_angle"], label="Head Tilt Angle")
plt.title(f"{title} - Head Tilt Angle Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Head Tilt Angle (degrees)")
plt.legend()
plt.grid(True)
plt.show()

# Plotting: Histogram of fatigue factor
plt.figure(figsize=(10, 5))
sns.histplot(df["fatigue_factor"], bins=20, kde=True, color="purple")
plt.title(f"{title} - Distribution of Fatigue Factor")
plt.xlabel("Fatigue Factor")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Correlation heatmap
correlation_matrix = df[numeric_columns].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title(f"{title} - Correlation Heatmap of Numeric Attributes")
plt.show()
