#!/usr/bin/env python3
"""
Plot a bar chart comparing the performance of the three models.
Run from project root: python plot_comparison.py
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

RESULTS_DIR = "results"
METRICS_FILE = os.path.join(RESULTS_DIR, "all_metrics.csv")

def plot_metrics(metrics_df: pd.DataFrame):
    """Generate and save a bar plot of the model metrics."""
    # Select and melt the dataframe
    metrics_to_plot = metrics_df[["model", "accuracy", "precision", "recall", "f1_weighted"]]
    melted_metrics = metrics_to_plot.melt(id_vars="model", var_name="Metric", value_name="Score")

    # Create the plot
    plt.figure(figsize=(12, 7))
    sns.barplot(x="model", y="Score", hue="Metric", data=melted_metrics)
    plt.title("Model Performance Comparison")
    plt.ylim(0.7, 1.0)
    plt.legend(title="Metric")
    
    # Save the plot
    plot_path = os.path.join(RESULTS_DIR, "model_comparison_chart.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Comparison chart saved to {plot_path}")
    plt.close()

def main():
    """Run the comparison script and plot the results."""
    # Ensure the metrics file is up to date
    print("Running model comparison to generate latest metrics...")
    subprocess.run(["./venv/bin/python3", "compare_models.py"], check=True)
    
    # Check if the metrics file exists
    if not os.path.exists(METRICS_FILE):
        print(f"Error: Metrics file not found at {METRICS_FILE}")
        print("Please ensure compare_models.py runs successfully.")
        return

    # Read the metrics and plot
    metrics_df = pd.read_csv(METRICS_FILE)
    plot_metrics(metrics_df)

if __name__ == "__main__":
    main()
