
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.preprocess import sliding_windows
from src.features import run_feature_extraction
from sklearn.utils.multiclass import unique_labels

# --- Configuration ---
MODELS_DIR = "results"
RESULTS_DIR = "results"
WINDOW_SIZE = 50
STRIDE = 25
MODEL_NAMES = ["svm", "knn", "rf"]

DATASETS = {
    "original": "data/processed/exercise_data.csv",
    "dev": "data/processed/dev_exercise_data.csv"
}

def plot_confusion_matrix(cm, labels, model_name, dataset_name):
    """Generates and saves a confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(max(12, len(labels) / 1.5), max(10, len(labels) / 2)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name.upper()} on {dataset_name.capitalize()} Data")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    output_path = os.path.join(RESULTS_DIR, f"cm_{dataset_name}_{model_name}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"    -> Saved confusion matrix to: {output_path}")

def plot_f1_comparison(all_metrics, dataset_name):
    """Generates and saves a grouped bar chart for F1 scores across all models."""
    # Convert metrics dict to a plottable DataFrame
    plot_data = []
    for model, metrics in all_metrics.items():
        for label, scores in metrics.items():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                plot_data.append({
                    'class': label,
                    'model': model.upper(),
                    'f1-score': scores['f1-score']
                })
    
    if not plot_data:
        print("    -> No per-class metrics found to plot.")
        return

    df = pd.DataFrame(plot_data)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.barplot(data=df, x='class', y='f1-score', hue='model', ax=ax)
    
    ax.set_title(f"F1-Score Comparison on {dataset_name.capitalize()} Data")
    ax.set_xlabel("Class (Exercise-Arm)")
    ax.set_ylabel("F1-Score")
    ax.legend(title="Model")
    plt.xticks(rotation=45, ha="right")
    
    output_path = os.path.join(RESULTS_DIR, f"f1_comparison_{dataset_name}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  -> Saved F1 comparison chart to: {output_path}")


def run_full_evaluation(dataset_name, file_path):
    """
    Runs a full evaluation pipeline on a given dataset, generating detailed visualizations.
    """
    print("-" * 80)
    print(f"Running evaluation on: {dataset_name.capitalize()} dataset")
    print("-" * 80)

    # 1. Load Data and Create Combined Class
    if not os.path.exists(file_path):
        print(f"  -> File not found: {file_path}. Skipping.")
        return
    
    df = pd.read_csv(file_path)

    if 'subject_id' not in df.columns and 'subject' in df.columns:
        df.rename(columns={'subject': 'subject_id'}, inplace=True)
    
    # Standardize arm label and create composite class
    if df['arm'].dtype == 'object':
        df['arm_std'] = df['arm'].str.upper().str[0] # Takes 'left' -> 'L', 'right' -> 'R'
    else: # Handles cases where it might be loaded as non-string
        df['arm_std'] = df['arm'].astype(str).str.upper().str[0]
        
    df['class'] = df['exercise'] + '-' + df['arm_std']
    
    print(f"  -> Loaded {len(df)} rows. Found {len(df['class'].unique())} unique classes.")

    # 2. Preprocessing (Windowing and Feature Extraction)
    try:
        windows, labels_class, _ = sliding_windows(df, window_size=WINDOW_SIZE, stride=STRIDE)
        if not windows:
            print("  -> No windows were generated after preprocessing. Skipping evaluation.")
            return
        X, y, _, _ = run_feature_extraction(windows, labels_class, labels_subject=[])
        print(f"  -> Preprocessing complete. Feature shape: {X.shape}")
    except Exception as e:
        print(f"  -> Could not preprocess data: {e}")
        return

    # 3. Evaluate Models
    all_model_metrics = {}

    for model_name in MODEL_NAMES:
        print(f"\\n  - Evaluating model: {model_name.upper()}...")
        model_path = os.path.join(MODELS_DIR, f"model_{model_name}.joblib")
        if not os.path.exists(model_path):
            print(f"    -> Model file not found: {model_path}. Skipping.")
            continue

        model = joblib.load(model_path)
        y_pred = model.predict(X)

        # Dynamically determine the union of labels present in true and predicted arrays
        cm_labels = unique_labels(y, y_pred)

        # Generate and Plot Confusion Matrix
        cm = confusion_matrix(y, y_pred, labels=cm_labels)
        plot_confusion_matrix(cm, cm_labels, model_name, dataset_name)
        
        # Store metrics for combined F1 plot
        report = classification_report(y, y_pred, labels=cm_labels, output_dict=True, zero_division=0)
        all_model_metrics[model_name] = report

    # 4. Generate Combined F1-Score Plot
    if all_model_metrics:
        plot_f1_comparison(all_model_metrics, dataset_name)


if __name__ == "__main__":
    for name, path in DATASETS.items():
        run_full_evaluation(name, path)
    print("\\nFull evaluation complete.")
