#!/usr/bin/env python3
"""
End-to-end Random Forest training: load data -> preprocess -> features -> train -> save results.
Run from project root: python run_train_rf.py
"""
import os
import sys

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Config (can be overridden by env or CLI later)
WINDOW_SIZE = 50
STRIDE = 25
TEST_SUBJECT_FRACTION = 0.2
RANDOM_STATE = 42
PROCESSED_PATH = "data/processed/exercise_data.csv"
FEATURE_DIR = "data/feature"
FEATURE_PATH = "data/feature/windowed_features.joblib"
RESULTS_DIR = "results"


def main() -> None:
    print("Step 1: Load raw data...")
    from src.load_data import load_and_process_data
    load_and_process_data()
    print("Step 2: Sliding-window segmentation...")
    from src.preprocess import run_preprocess
    windows, labels_exercise, labels_subject = run_preprocess(
        processed_path=PROCESSED_PATH,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )
    print(f"  -> {len(windows)} windows")
    print("Step 3: Feature extraction...")
    from src.features import run_feature_extraction, save_features
    X, y, subject_ids, feature_names = run_feature_extraction(
        windows, labels_exercise, labels_subject,
    )
    os.makedirs(FEATURE_DIR, exist_ok=True)
    save_features(X, y, subject_ids, feature_names, out_dir=FEATURE_DIR)
    print(f"  -> X shape {X.shape}, saved to {FEATURE_DIR}/")
    print("Step 4: Train Random Forest (subject-independent split)...")
    from src.train import run_train
    metrics = run_train(
        feature_path=FEATURE_PATH,
        results_dir=RESULTS_DIR,
        test_subject_fraction=TEST_SUBJECT_FRACTION,
        random_state=RANDOM_STATE,
        save_model=True,
    )
    print("Done.")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"  Results in {RESULTS_DIR}/ (confusion_matrix_rf.png, metrics_rf.csv, model_rf.joblib)")


if __name__ == "__main__":
    main()
