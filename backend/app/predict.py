from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


FEATURES = [
    "bikes_now","docks_now",
    "bikes_delta_10min","docks_delta_10min",
    "bikes_delta_30min","docks_delta_30min",
    "pct_bikes","pct_docks",
    "hour","dayofweek","is_weekend"
]

NUM_COLS = [
    "bikes_now","docks_now",
    "bikes_delta_10min","docks_delta_10min",
    "bikes_delta_30min","docks_delta_30min",
    "pct_bikes","pct_docks",
]

CAT_COLS = ["hour","dayofweek","is_weekend"]


def _build_pipeline():
    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, NUM_COLS),
            ("cat", categorical, CAT_COLS),
        ]
    )

    clf = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",  # IMPORTANT for FULL_30
        n_jobs=None
    )

    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def _best_f1_threshold(y_true, p):
    precision, recall, thresholds = precision_recall_curve(y_true, p)
    f1 = (2 * precision * recall) / (precision + recall + 1e-9)
    best_idx = int(np.nanargmax(f1))
    # thresholds is shorter by 1 than precision/recall; guard index
    if len(thresholds) == 0:
        return 0.5
    thr_idx = max(best_idx - 1, 0)
    return float(thresholds[thr_idx])


def train_models(parquet_path: str, out_dir: str):
    df = pd.read_parquet(parquet_path)

    X = df[FEATURES].copy()
    y_empty = df["empty_30"].astype(int).values
    y_full = df["full_30"].astype(int).values

    results = {}

    def train_one(name, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        model = _build_pipeline()
        model.fit(X_train, y_train)

        p = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, p)
        pr = average_precision_score(y_test, p)
        thr = _best_f1_threshold(y_test, p)

        return model, {"roc_auc": float(roc), "pr_auc": float(pr), "threshold": float(thr)}

    empty_model, empty_stats = train_one("empty_30", y_empty)
    full_model, full_stats = train_one("full_30", y_full)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(empty_model, out / "empty_30_model.joblib")
    joblib.dump(full_model, out / "full_30_model.joblib")

    thresholds = {
        "empty_30_threshold": empty_stats["threshold"],
        "full_30_threshold": full_stats["threshold"],
        "empty_30_metrics": empty_stats,
        "full_30_metrics": full_stats,
        "features": FEATURES,
    }
    joblib.dump(thresholds, out / "thresholds.joblib")

    results["empty_30"] = empty_stats
    results["full_30"] = full_stats
    return results

FEATURES_ORDER = FEATURES

def build_feature_df(rows: list[dict]) -> pd.DataFrame:
    """
    Takes a list of feature dicts (one per station) and returns a DataFrame
    with columns in the exact order the model expects.
    Missing columns are filled with 0.
    """
    df = pd.DataFrame(rows)
    for col in FEATURES_ORDER:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURES_ORDER].copy()


def predict_proba(model_path: str, X: pd.DataFrame) -> list[float]:
    """
    Loads a saved sklearn pipeline model and returns probability of class=1
    for each row.
    """
    model = joblib.load(model_path)
    p = model.predict_proba(X)[:, 1]
    return [float(x) for x in p]
