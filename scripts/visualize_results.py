# scripts/visualize_results.py

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import argparse
import os
import time

def visualize_emotions(analysis_csv, trends_image):
    start_time = time.time()
    # Read only necessary columns
    df = pd.read_csv(analysis_csv, usecols=['frame', 'emotion', 'driver_confidence_level'])
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    print(f"DataFrame shape: {df.shape}")

    # Optimize frame number extraction
    start_time = time.time()
    df['frame_number'] = df['frame'].str.extract(r'_(\d+)\.jpg').astype(int)
    df['time_seconds'] = df['frame_number']  # fps=1, so frame number equals seconds
    print(f"Frame numbers processed in {time.time() - start_time:.2f} seconds")

    # Convert to appropriate data types
    df['emotion'] = df['emotion'].astype('category')
    df['time_seconds'] = pd.to_numeric(df['time_seconds'], downcast='integer')
    df['driver_confidence_level'] = pd.to_numeric(df['driver_confidence_level'], downcast='float')

    # Group and time the operation
    start_time = time.time()
    emotion_counts = df.groupby(['time_seconds', 'emotion'], observed=False).size().unstack(fill_value=0)
    print(f"Groupby operation completed in {time.time() - start_time:.2f} seconds")

    # Optionally reduce the number of emotions plotted
    top_emotions = df['emotion'].value_counts().nlargest(3).index
    emotion_counts = emotion_counts[top_emotions]

    # Plotting
    start_time = time.time()
    fig, ax1 = plt.subplots(figsize=(14, 7))
    emotion_counts.plot(kind='line', ax=ax1)
    print(f"Emotion counts plotted in {time.time() - start_time:.2f} seconds")

    ax1.set_title('Driver Emotion Analysis Over Time')
    ax1.set_xlabel('Time (Seconds)')
    ax1.set_ylabel('Emotion Counts')
    ax1.legend(title='Emotions')

    # Plot driver confidence level over time
    start_time = time.time()
    ax2 = ax1.twinx()
    confidence_over_time = df.groupby('time_seconds')['driver_confidence_level'].mean()
    confidence_over_time.plot(kind='line', color='black', linestyle='--', ax=ax2)
    print(f"Driver confidence level plotted in {time.time() - start_time:.2f} seconds")

    ax2.set_ylabel('Driver Confidence Level')
    ax2.legend(['Confidence Level'], loc='upper right')

    # Save the plot
    start_time = time.time()
    plt.savefig(trends_image)
    print(f"Plot saved in {time.time() - start_time:.2f} seconds")
    print(f"Emotion trends saved as {trends_image}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize emotion analysis results.")
    parser.add_argument('--analysis_csv', type=str, required=True, help='CSV file with emotion analysis.')
    parser.add_argument('--trends_image', type=str, required=True, help='Path to save the emotion trends image.')
    args = parser.parse_args()
    visualize_emotions(args.analysis_csv, args.trends_image)
