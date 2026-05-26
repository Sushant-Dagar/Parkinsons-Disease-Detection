"""
train.py — Parkinson's Disease Detection
========================================

Trains an SVM classifier on the UCI Parkinson's voice dataset.

Pipeline:
    1. Download the UCI dataset if not present locally
    2. Drop identifier column ('name'), split features/target
    3. Stratified 80/20 train/test split, then StandardScaler on features
    4. GridSearchCV over SVM hyperparameters with 5-fold stratified CV (scoring=F1)
    5. Train baseline models (Logistic Regression, Random Forest) for comparison
    6. Evaluate best model on held-out test set, save metrics to metrics.json
    7. Persist trained pipeline to model.pkl

Run:
    python train.py
"""

import json
import os
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------- config ----------
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
DATA_PATH = Path("parkinsons.csv")
MODEL_PATH = Path("model.pkl")
METRICS_PATH = Path("metrics.json")
SEED = 42
TEST_SIZE = 0.20

# ---------- data ----------
def load_data() -> pd.DataFrame:
    """Download the UCI Parkinson's dataset if needed, then load it."""
    if not DATA_PATH.exists():
        print(f"Downloading dataset to {DATA_PATH} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    # 'name' is the recording identifier (e.g. phon_R01_S01_1) — drop it.
    if "name" in df.columns:
        df = df.drop(columns=["name"])
    return df


def split_xy(df: pd.DataFrame):
    y = df["status"].astype(int).values            # 1 = Parkinson's, 0 = healthy
    X = df.drop(columns=["status"]).values
    feature_names = df.drop(columns=["status"]).columns.tolist()
    return X, y, feature_names


# ---------- training ----------
def grid_search_svm(X_train, y_train) -> GridSearchCV:
    """Tune SVM hyperparameters with stratified 5-fold CV."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(probability=True, random_state=SEED)),
    ])
    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__kernel": ["linear", "rbf", "poly"],
        "svm__gamma": ["scale", "auto"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(pipe, param_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    return gs


def baseline_scores(X_train, y_train) -> dict[str, float]:
    """5-fold CV F1 for baseline models, for honest model-selection reporting."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    baselines = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=SEED)),
        ]),
    }
    scores = {}
    for name, model in baselines.items():
        cv_scores = cross_val_score(model, X_train, y_train, scoring="f1", cv=cv, n_jobs=-1)
        scores[name] = {
            "cv_f1_mean": round(float(cv_scores.mean()), 4),
            "cv_f1_std":  round(float(cv_scores.std()),  4),
        }
    return scores


# ---------- evaluation ----------
def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {
        "f1":         round(float(f1_score(y_test, y_pred)),        4),
        "precision":  round(float(precision_score(y_test, y_pred)), 4),
        "recall":     round(float(recall_score(y_test, y_pred)),    4),
        "roc_auc":    round(float(roc_auc_score(y_test, y_proba)),  4) if y_proba is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    return metrics


# ---------- main ----------
def main():
    df = load_data()
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1] - 1} features")
    print(f"Class balance — healthy: {(df['status'] == 0).sum()}, parkinsons: {(df['status'] == 1).sum()}")

    X, y, feature_names = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    print("\n=== Grid-searching SVM ===")
    gs = grid_search_svm(X_train, y_train)
    print(f"Best params: {gs.best_params_}")
    print(f"Best CV F1:  {gs.best_score_:.4f}")

    print("\n=== Baselines (5-fold CV F1) ===")
    baselines = baseline_scores(X_train, y_train)
    for name, s in baselines.items():
        print(f"  {name:22s}  F1 = {s['cv_f1_mean']:.4f} ± {s['cv_f1_std']:.4f}")

    print("\n=== Held-out test set (SVM) ===")
    test_metrics = evaluate(gs.best_estimator_, X_test, y_test)
    print(f"  F1        = {test_metrics['f1']}")
    print(f"  Precision = {test_metrics['precision']}")
    print(f"  Recall    = {test_metrics['recall']}")
    print(f"  ROC-AUC   = {test_metrics['roc_auc']}")
    print(f"  Confusion matrix:\n{np.array(test_metrics['confusion_matrix'])}")

    # persist
    joblib.dump(gs.best_estimator_, MODEL_PATH)
    print(f"\nSaved model → {MODEL_PATH}")

    summary = {
        "dataset": "UCI Parkinson's Voice (N=195, 22 features)",
        "test_size": TEST_SIZE,
        "seed": SEED,
        "best_params": {k: str(v) for k, v in gs.best_params_.items()},
        "cv_f1_best": round(float(gs.best_score_), 4),
        "baselines_cv_f1": baselines,
        "test": {
            "f1":        test_metrics["f1"],
            "precision": test_metrics["precision"],
            "recall":    test_metrics["recall"],
            "roc_auc":   test_metrics["roc_auc"],
            "confusion_matrix": test_metrics["confusion_matrix"],
        },
        "feature_names": feature_names,
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Saved metrics → {METRICS_PATH}")


if __name__ == "__main__":
    main()
