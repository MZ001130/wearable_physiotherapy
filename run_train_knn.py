#!/usr/bin/env python3
"""
End-to-end k-NN training: load data -> preprocess -> features -> train -> save results.
Run from project root: python run_train_knn.py
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from typing import Union

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Config (can be overridden by env or CLI later)
WINDOW_SIZE = 50
STRIDE = 25
TEST_SUBJECT_FRACTION = 0.2
N_NEIGHBORS = 5
RANDOM_STATE = 42
PROCESSED_PATH = "data/processed/exercise_data.csv"
FEATURE_DIR = "data/feature"
FEATURE_PATH = "data/feature/windowed_features.joblib"
RESULTS_DIR = "results"


def subject_independent_split(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    test_subject_fraction: float = TEST_SUBJECT_FRACTION,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list]:
    """Split data by subject so that no subject appears in both train and test."""
    rng = np.random.default_rng(random_state)
    subjects = np.unique(subject_ids)
    n_test = max(1, int(round(len(subjects) * test_subject_fraction)))
    n_test = min(n_test, len(subjects) - 1)
    perm = rng.permutation(subjects)
    test_subjects = list(perm[:n_test])
    train_subjects = list(perm[n_test:])

    train_mask = np.isin(subject_ids, train_subjects)
    test_mask = np.isin(subject_ids, test_subjects)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    return X_train, X_test, y_train, y_test, train_subjects, test_subjects


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    clf: object,
    class_labels: Union[np.ndarray, list, None] = None,
) -> tuple[object, np.ndarray, dict, np.ndarray]:
    """Train classifier and compute accuracy, F1, confusion matrix."""
    if class_labels is None:
        class_labels = sorted(set(y_train).union(set(y_test)))

    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "training_time": training_time,
    }
    return clf, y_pred, metrics, cm


def save_results(
    metrics: dict,
    confusion_mat: np.ndarray,
    class_labels: np.ndarray,
    train_subjects: list,
    test_subjects: list,
    results_dir: str = RESULTS_DIR,
    save_model: bool = True,
    model: Union[object, None] = None,
    suffix: str = "_knn",
) -> None:
    """Write metrics CSV, confusion matrix plot, and optionally the model."""
    os.makedirs(results_dir, exist_ok=True)

    m = pd.DataFrame([{
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
        "training_time": metrics["training_time"],
        "n_train": metrics["n_train"],
        "n_test": metrics["n_test"],
        "train_subjects": ",".join(str(s) for s in train_subjects),
        "test_subjects": ",".join(str(s) for s in test_subjects),
    }])
    m.to_csv(os.path.join(results_dir, f"metrics{suffix}.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    labels = list(class_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_mat, xticklabels=labels, yticklabels=labels, annot=True, fmt="d", ax=ax, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {suffix.strip('_').upper()} (subject-independent test)")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f"confusion_matrix{suffix}.png"), dpi=150)
    plt.close()

    if save_model and model is not None:
        joblib.dump(model, os.path.join(results_dir, f"model{suffix}.joblib"))


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
    print("Step 4: Train k-NN (subject-independent split)...")
    from src.features import load_features
    X, y, subject_ids, _ = load_features(FEATURE_PATH)

    X_train, X_test, y_train, y_test, train_subjects, test_subjects = subject_independent_split(
        X, y, subject_ids,
        test_subject_fraction=TEST_SUBJECT_FRACTION,
        random_state=RANDOM_STATE,
    )
    class_labels = np.array(sorted(set(y_train).union(set(y_test))))

    clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=N_NEIGHBORS))

    clf, y_pred, metrics, cm = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        clf=clf,
        class_labels=class_labels,
    )
    save_results(
        metrics, cm, class_labels,
        train_subjects, test_subjects,
        results_dir=RESULTS_DIR,
        save_model=True,
        model=clf,
        suffix="_knn",
    )
    print("Done.")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"  Training time: {metrics['training_time']:.4f}s")
    print(f"  Results in {RESULTS_DIR}/ (confusion_matrix_knn.png, metrics_knn.csv, model_knn.joblib)")


if __name__ == "__main__":
    main()
