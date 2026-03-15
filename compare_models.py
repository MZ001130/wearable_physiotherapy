#!/usr/bin/env python3
"""
Compare the performance of the three models (RF, SVM, k-NN).
Run from project root: python compare_models.py
"""
import os
import pandas as pd
import subprocess

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

RESULTS_DIR = "results"

def run_all_training_scripts():
    """Run all training scripts to generate the latest results."""
    scripts = ["run_train_rf.py", "run_train_svm.py", "run_train_knn.py"]
    for script in scripts:
        print(f"Running {script}...")
        subprocess.run(["./venv/bin/python3", script], check=True)
        print("-" * 30)

def combine_metrics():
    """Combine metrics from all models into a single DataFrame."""
    metrics_files = {
        "Random Forest": os.path.join(RESULTS_DIR, "metrics_rf.csv"),
        "SVM": os.path.join(RESULTS_DIR, "metrics_svm.csv"),
        "k-NN": os.path.join(RESULTS_DIR, "metrics_knn.csv"),
    }

    all_metrics = []
    for model_name, path in metrics_files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["model"] = model_name
            all_metrics.append(df)
        else:
            print(f"Warning: Metrics file not found for {model_name} at {path}")

    if not all_metrics:
        print("No metrics files found. Please run the training scripts first.")
        return None

    combined = pd.concat(all_metrics, ignore_index=True)
    return combined

def main():
    """Run all training scripts and print a comparison table."""
    run_all_training_scripts()
    metrics = combine_metrics()
    if metrics is not None:
        # Save combined metrics
        metrics.to_csv(os.path.join(RESULTS_DIR, "all_metrics.csv"), index=False)
        print(f"\nCombined metrics saved to {os.path.join(RESULTS_DIR, 'all_metrics.csv')}")

        # Select and rename columns for the final table
        table = metrics[["model", "accuracy", "precision", "recall", "f1_weighted", "training_time"]].copy()
        table.rename(columns={
            "model": "Model",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1_weighted": "F1-Score",
            "training_time": "Time to Train (s)",
        }, inplace=True)

        print("\n--- Model Comparison ---")
        print(table.to_markdown(index=False))

if __name__ == "__main__":
    main()
