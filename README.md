# Wearable Physiotherapy — Exercise Recognition from IMU Data

Classify shoulder physiotherapy exercises from Apple Watch IMU data (SPAR dataset) using feature-based machine learning (Random Forest, with k-NN and SVM to be added). Evaluation uses **subject-independent** train/test splits.

## Python environment

1. **Create and activate a virtual environment** (from project root):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Pipeline (Random Forest training)

Run in order:

1. **Load raw data** → writes `data/processed/exercise_data.csv`:
   ```bash
   python src/load_data.py
   ```

2. **Preprocess** (sliding-window segmentation) → used by feature extraction:
   ```bash
   python src/preprocess.py
   ```
   (Or skip: the training entrypoint runs preprocessing internally.)

3. **Train Random Forest** (preprocess → features → subject-independent split → train → evaluate):
   ```bash
   python run_train_rf.py
   ```
   This creates `data/feature/` and `results/` if needed, saves features, trains the model, and writes:
   - `results/confusion_matrix_rf.png`
   - `results/metrics_rf.csv`
   - `results/model_rf.joblib` (optional)

To run steps individually: use `src/preprocess.py`, `src/features.py`, and `src/train.py` (see docstrings for usage).

## Data layout

- `data/raw/` — SPAR dataset CSVs (S1–S20, E0–E6, L/R).
- `data/processed/` — combined `exercise_data.csv` (output of `load_data.py`).
- `data/feature/` — windowed feature matrix and labels (output of feature extraction).
- `results/` — metrics, confusion matrix plot, saved model.

## Reproducibility

Use fixed `random_state` in `run_train_rf.py` (and in `src/train.py`) for reproducible subject splits and Random Forest training.
