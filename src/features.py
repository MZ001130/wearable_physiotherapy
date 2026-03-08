"""
Feature extraction from windowed IMU segments.
Per-channel statistics and optional magnitude features.
"""
import os
import numpy as np
import pandas as pd
import joblib

# Default signal column order (must match preprocess)
SIGNAL_COLS = ["ax", "ay", "az", "wx", "wy", "wz"]

# Feature presets for sensitivity analysis
FEATURE_SET_STATISTICS = ["mean", "std", "min", "max", "range"]
FEATURE_SET_DEFAULT = FEATURE_SET_STATISTICS  # extend with "magnitude_mean", "magnitude_std" if needed


def _compute_statistics(win: np.ndarray, channel_names: list[str], prefix: str) -> dict[str, float]:
    """Compute mean, std, min, max, range per channel. win shape (window_size, n_channels)."""
    out = {}
    for j, name in enumerate(channel_names):
        col = win[:, j]
        out[f"{prefix}_{name}_mean"] = float(np.mean(col))
        out[f"{prefix}_{name}_std"] = float(np.std(col))
        out[f"{prefix}_{name}_min"] = float(np.min(col))
        out[f"{prefix}_{name}_max"] = float(np.max(col))
        out[f"{prefix}_{name}_range"] = float(np.max(col) - np.min(col))
    return out


def _compute_magnitude_features(win: np.ndarray, accel_idx: list[int], gyro_idx: list[int]) -> dict[str, float]:
    """Magnitude mean/std for accel and gyro. win shape (window_size, n_channels)."""
    out = {}
    if accel_idx:
        accel = win[:, accel_idx]
        mag_acc = np.sqrt(np.sum(accel ** 2, axis=1))
        out["magnitude_accel_mean"] = float(np.mean(mag_acc))
        out["magnitude_accel_std"] = float(np.std(mag_acc))
    if gyro_idx:
        gyro = win[:, gyro_idx]
        mag_gyro = np.sqrt(np.sum(gyro ** 2, axis=1))
        out["magnitude_gyro_mean"] = float(np.mean(mag_gyro))
        out["magnitude_gyro_std"] = float(np.std(mag_gyro))
    return out


def extract_features(
    windows: list[np.ndarray],
    channel_names: list[str] | None = None,
    feature_set: list[str] | None = None,
    include_magnitude: bool = True,
) -> np.ndarray:
    """
    Extract feature vector for each window.

    Parameters
    ----------
    windows : list of np.ndarray
        Each (window_size, n_channels) in order ax, ay, az, wx, wy, wz.
    channel_names : list[str] or None
        Names for each channel. Default: SIGNAL_COLS.
    feature_set : list[str] or None
        Which stats to compute. Default: mean, std, min, max, range.
    include_magnitude : bool
        If True, add magnitude mean/std for accel and gyro.

    Returns
    -------
    X : np.ndarray shape (n_windows, n_features)
    """
    if channel_names is None:
        channel_names = SIGNAL_COLS.copy()
    if feature_set is None:
        feature_set = FEATURE_SET_DEFAULT.copy()

    rows = []
    for win in windows:
        row = _compute_statistics(win, channel_names, "ch")
        if include_magnitude:
            accel_idx = [0, 1, 2]
            gyro_idx = [3, 4, 5]
            row.update(_compute_magnitude_features(win, accel_idx, gyro_idx))
        rows.append(row)

    return pd.DataFrame(rows).values.astype(np.float64)


def get_feature_names(
    channel_names: list[str] | None = None,
    include_magnitude: bool = True,
) -> list[str]:
    """Return ordered feature names matching extract_features output."""
    if channel_names is None:
        channel_names = SIGNAL_COLS.copy()
    names = []
    for stat in FEATURE_SET_DEFAULT:
        for ch in channel_names:
            names.append(f"ch_{ch}_{stat}")
    if include_magnitude:
        names.extend(["magnitude_accel_mean", "magnitude_accel_std", "magnitude_gyro_mean", "magnitude_gyro_std"])
    return names


def run_feature_extraction(
    windows: list[np.ndarray],
    labels_exercise: list[str],
    labels_subject: list[str],
    include_magnitude: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Extract features and return X, y (exercise), subject_ids, feature_names.
    """
    X = extract_features(windows, include_magnitude=include_magnitude)
    feature_names = get_feature_names(include_magnitude=include_magnitude)
    y = np.array(labels_exercise)
    subject_ids = np.array(labels_subject)
    return X, y, subject_ids, feature_names


def save_features(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    feature_names: list[str],
    out_dir: str = "data/feature",
    base_name: str = "windowed_features",
) -> str:
    """
    Save feature matrix and metadata to out_dir.
    Writes base_name.joblib (X, y, subject_ids, feature_names) and base_name_meta.csv for readability.
    Returns path to the .joblib file.
    """
    os.makedirs(out_dir, exist_ok=True)
    path_joblib = os.path.join(out_dir, f"{base_name}.joblib")
    path_csv = os.path.join(out_dir, f"{base_name}_meta.csv")
    joblib.dump({"X": X, "y": y, "subject_ids": subject_ids, "feature_names": feature_names}, path_joblib)
    meta = pd.DataFrame({"exercise": y, "subject_id": subject_ids})
    meta.to_csv(path_csv, index=False)
    # Optionally save full feature matrix with labels for debugging
    full = pd.DataFrame(X, columns=feature_names)
    full["exercise"] = y
    full["subject_id"] = subject_ids
    full.to_csv(os.path.join(out_dir, f"{base_name}.csv"), index=False)
    return path_joblib


def load_features(path: str = "data/feature/windowed_features.joblib") -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load X, y, subject_ids, feature_names from a .joblib file."""
    data = joblib.load(path)
    return data["X"], data["y"], data["subject_ids"], data["feature_names"]


if __name__ == "__main__":
    # Run from project root: python src/features.py
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocess import run_preprocess

    windows, labels_ex, labels_subj = run_preprocess()
    X, y, subject_ids, names = run_feature_extraction(windows, labels_ex, labels_subj)
    path = save_features(X, y, subject_ids, names)
    print(f"Features shape: {X.shape}, saved to {path}")
