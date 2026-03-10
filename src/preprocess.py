"""
Sliding-window segmentation for IMU time series.
Groups by (subject_id, exercise) so segments do not cross trials.
"""
import os
import pandas as pd
import numpy as np
from typing import Union


def load_processed_data(processed_path: str = "data/processed/exercise_data.csv") -> pd.DataFrame:
    """Load combined exercise data from load_data.py output."""
    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. Run src/load_data.py first."
        )
    return pd.read_csv(processed_path)


def sliding_windows(
    df: pd.DataFrame,
    window_size: int = 50,
    stride: int = 25,
    signal_cols: Union[list[str], None] = None,
) -> tuple[list[np.ndarray], list[str], list[str]]:
    """
    Extract sliding windows per (subject_id, exercise) trial.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain signal columns and 'subject_id', 'exercise'.
    window_size : int
        Number of samples per window.
    stride : int
        Step size between consecutive windows (overlap = window_size - stride).
    signal_cols : list[str] or None
        Columns to use as signals. Default: ["ax", "ay", "az", "wx", "wy", "wz"].

    Returns
    -------
    windows : list of np.ndarray
        Each element has shape (window_size, n_channels).
    labels_exercise : list[str]
        Exercise label per window.
    labels_subject : list[str]
        Subject ID per window.
    """
    if signal_cols is None:
        signal_cols = ["ax", "ay", "az", "wx", "wy", "wz"]
    for c in signal_cols + ["subject_id", "exercise"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    windows = []
    labels_exercise = []
    labels_subject = []

    grouped = df.groupby(["subject_id", "exercise"], group_keys=False)
    for (subject_id, exercise), grp in grouped:
        block = grp[signal_cols].values.astype(np.float64)
        n = len(block)
        if n < window_size:
            continue
        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            windows.append(block[start:end].copy())
            labels_exercise.append(exercise)
            labels_subject.append(subject_id)

    return windows, labels_exercise, labels_subject


def run_preprocess(
    processed_path: str = "data/processed/exercise_data.csv",
    window_size: int = 50,
    stride: int = 25,
) -> tuple[list[np.ndarray], list[str], list[str]]:
    """
    Load processed data and return sliding-window segments.
    Use this from an entrypoint or from features.py.
    """
    df = load_processed_data(processed_path)
    return sliding_windows(df, window_size=window_size, stride=stride)


if __name__ == "__main__":
    import sys
    # Run from project root: python src/preprocess.py
    processed = "data/processed/exercise_data.csv"
    window_size = 50
    stride = 25
    if len(sys.argv) > 1:
        window_size = int(sys.argv[1])
    if len(sys.argv) > 2:
        stride = int(sys.argv[2])
    windows, ex, subj = run_preprocess(processed, window_size=window_size, stride=stride)
    print(f"Preprocessed: {len(windows)} windows (window_size={window_size}, stride={stride})")
    print(f"Exercises: {sorted(set(ex))}, Subjects: {sorted(set(subj))}")
