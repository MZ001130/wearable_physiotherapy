"""
Random Forest training with subject-independent train/test split.
Loads features, splits by subject_id, trains RF, evaluates and saves metrics and confusion matrix.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Default paths (relative to project root)
DEFAULT_FEATURE_PATH = "data/feature/windowed_features.joblib"
DEFAULT_RESULTS_DIR = "results"
RANDOM_STATE = 42
N_ESTIMATORS = 100
TEST_SUBJECT_FRACTION = 0.2  # Hold out 20% of subjects for test


def subject_independent_split(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    test_subject_fraction: float = TEST_SUBJECT_FRACTION,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list]:
    """
    Split data by subject so that no subject appears in both train and test.

    Parameters
    ----------
    X, y, subject_ids : arrays
        Feature matrix, labels, and subject id per sample.
    test_subject_fraction : float
        Fraction of unique subjects to hold out for test (e.g. 0.2 => 20%).
    random_state : int
        For reproducible subject selection.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    train_subjects, test_subjects : list
        Subject IDs in each set (for documentation).
    """
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
    n_estimators: int = N_ESTIMATORS,
    random_state: int = RANDOM_STATE,
    class_labels: np.ndarray | list | None = None,
) -> tuple[RandomForestClassifier, np.ndarray, dict, np.ndarray]:
    """
    Train Random Forest and compute accuracy, F1, confusion matrix.
    Returns fitted model, y_pred, metrics dict, confusion matrix.
    """
    if class_labels is None:
        class_labels = sorted(set(y_train) | set(y_test))
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }
    return clf, y_pred, metrics, cm


def save_results(
    metrics: dict,
    confusion_mat: np.ndarray,
    class_labels: np.ndarray,
    train_subjects: list,
    test_subjects: list,
    results_dir: str = DEFAULT_RESULTS_DIR,
    save_model: bool = True,
    model: RandomForestClassifier | None = None,
) -> None:
    """Write metrics CSV, confusion matrix plot, and optionally the model."""
    os.makedirs(results_dir, exist_ok=True)

    # Metrics table
    m = pd.DataFrame([{
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
        "n_train": metrics["n_train"],
        "n_test": metrics["n_test"],
        "train_subjects": ",".join(str(s) for s in train_subjects),
        "test_subjects": ",".join(str(s) for s in test_subjects),
    }])
    m.to_csv(os.path.join(results_dir, "metrics_rf.csv"), index=False)

    # Confusion matrix heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    labels = list(class_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_mat, xticklabels=labels, yticklabels=labels, annot=True, fmt="d", ax=ax, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Random Forest (subject-independent test)")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "confusion_matrix_rf.png"), dpi=150)
    plt.close()

    if save_model and model is not None:
        joblib.dump(model, os.path.join(results_dir, "model_rf.joblib"))


def run_train(
    feature_path: str = DEFAULT_FEATURE_PATH,
    results_dir: str = DEFAULT_RESULTS_DIR,
    test_subject_fraction: float = TEST_SUBJECT_FRACTION,
    n_estimators: int = N_ESTIMATORS,
    random_state: int = RANDOM_STATE,
    save_model: bool = True,
) -> dict:
    """
    Load features, perform subject-independent split, train RF, save results.
    Returns metrics dict.
    """
    from src.features import load_features  # noqa: F401
    X, y, subject_ids, _ = load_features(feature_path)

    X_train, X_test, y_train, y_test, train_subjects, test_subjects = subject_independent_split(
        X, y, subject_ids,
        test_subject_fraction=test_subject_fraction,
        random_state=random_state,
    )
    class_labels = np.array(sorted(set(y_train) | set(y_test)))
    clf, y_pred, metrics, cm = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        n_estimators=n_estimators,
        random_state=random_state,
        class_labels=class_labels,
    )
    save_results(
        metrics, cm, class_labels,
        train_subjects, test_subjects,
        results_dir=results_dir,
        save_model=save_model,
        model=clf,
    )
    return metrics


if __name__ == "__main__":
    # Run from project root: python src/train.py
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from features import load_features

    if not os.path.exists(DEFAULT_FEATURE_PATH):
        print(f"Feature file not found: {DEFAULT_FEATURE_PATH}. Run run_train_rf.py or features.py first.")
        sys.exit(1)

    X, y, subject_ids, _ = load_features(DEFAULT_FEATURE_PATH)
    X_train, X_test, y_train, y_test, train_s, test_s = subject_independent_split(X, y, subject_ids)
    class_labels = np.array(sorted(set(y_train) | set(y_test)))
    clf, y_pred, metrics, cm = train_and_evaluate(X_train, y_train, X_test, y_test, class_labels=class_labels)
    save_results(metrics, cm, class_labels, train_s, test_s, model=clf)
    print("Metrics:", metrics)
    print(f"Train subjects: {train_s}, Test subjects: {test_s}")
    print(f"Results saved to {DEFAULT_RESULTS_DIR}/")
