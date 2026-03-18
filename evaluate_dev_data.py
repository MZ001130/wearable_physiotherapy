
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.preprocess import sliding_windows
from src.features import run_feature_extraction

# --- Configuration ---
DEV_DATA_PATH = "data/processed/dev_exercise_data.csv"
MODELS_DIR = "results"
RESULTS_DIR = "results"
WINDOW_SIZE = 50
STRIDE = 25
MODEL_NAMES = ["svm", "knn", "rf"]

def evaluate_models_on_dev_data():
    """
    Loads dev data, preprocesses it, extracts features, and evaluates pre-trained models.
    Generates and saves a confusion matrix for each model.
    """
    print("Step 1: Loading and preprocessing dev data...")
    if not os.path.exists(DEV_DATA_PATH):
        print(f"Error: Dev data not found at {DEV_DATA_PATH}")
        return

    # Load data into memory
    df = pd.read_csv(DEV_DATA_PATH)

    # Handle potential column name mismatch for subject identifier
    if 'subject_id' not in df.columns and 'subject' in df.columns:
        print("  -> 'subject_id' column not found, renaming 'subject' to 'subject_id'.")
        df.rename(columns={'subject': 'subject_id'}, inplace=True)

    # Preprocess the data (windowing) directly from the dataframe
    try:
        windows, labels_exercise, _ = sliding_windows(
            df,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
        )
        print(f"  -> Created {len(windows)} windows from dev data.")
    except ValueError as e:
        print(f"Error during windowing process: {e}")
        return



    print("Step 2: Extracting features from dev data...")
    # Feature extraction
    X_dev, y_dev, _, feature_names = run_feature_extraction(
        windows, labels_exercise, labels_subject=[] # We don't need subject labels for this part
    )
    print(f"  -> Extracted features. Shape: {X_dev.shape}")

    # Get the class labels in the correct order
    class_labels = sorted(list(set(y_dev)))

    print("\nStep 3: Evaluating models and generating confusion matrices...")
    for model_name in MODEL_NAMES:
        model_path = os.path.join(MODELS_DIR, f"model_{model_name}.joblib")
        if not os.path.exists(model_path):
            print(f"  - WARNING: Model file not found for '{model_name}'. Skipping.")
            continue

        print(f"  - Evaluating model: {model_name.upper()}...")
        # Load the trained model
        model = joblib.load(model_path)

        # Make predictions
        y_pred = model.predict(X_dev)

        # Generate confusion matrix
        cm = confusion_matrix(y_dev, y_pred, labels=class_labels)

        # Plot and save confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        ax.set_title(f"Confusion Matrix - {model_name.upper()} on Dev Data")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        
        output_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name}_dev.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        
        print(f"    -> Saved confusion matrix to: {output_path}")

    print("\nEvaluation complete.")

if __name__ == "__main__":
    evaluate_models_on_dev_data()

