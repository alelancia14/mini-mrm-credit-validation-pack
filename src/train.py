from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "model_table.csv"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "logit_pd_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "outputs" / "logs" / "train_metadata.json"


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("config.yaml must contain a top-level mapping.")
    return raw


def run_train() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}. Run `python -m src.build_table` first."
        )

    config = _load_config(CONFIG_PATH)
    random_seed = int(config.get("random_seed", 42))
    test_size = float(config.get("test_size", 0.2))
    psi_bins = int(config.get("psi_bins", 10))

    df = pd.read_csv(INPUT_PATH)
    if "default" not in df.columns:
        raise ValueError("Expected target column `default` in model_table.csv.")

    X = df.drop(columns=["default"])
    y = df["default"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    numeric_features = list(X_train.select_dtypes(include="number").columns)
    categorical_features = [
        col for col in X_train.columns if col not in numeric_features
    ]

    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=200, solver="liblinear")),
        ]
    )
    pipeline.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    n_features_after_encoding = int(
        len(pipeline.named_steps["preprocess"].get_feature_names_out())
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "default_rate_train": float(y_train.mean()),
        "default_rate_test": float(y_test.mean()),
        "n_features_after_encoding": n_features_after_encoding,
        "random_seed": random_seed,
        "test_size": test_size,
        "psi_bins": psi_bins,
        "numeric_preprocessing": "passthrough",
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn_version,
        "duckdb_version": duckdb.__version__,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"rows_train={metadata['rows_train']}")
    print(f"rows_test={metadata['rows_test']}")
    print(f"default_rate_train={metadata['default_rate_train']:.4f}")
    print(f"default_rate_test={metadata['default_rate_test']:.4f}")
    print(f"n_features_after_encoding={n_features_after_encoding}")
    print(f"model_path={MODEL_PATH}")


if __name__ == "__main__":
    run_train()
