from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "credit_g.csv"
METADATA_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "logs" / "ingest_metadata.json"


def _to_default_binary(target: pd.Series) -> pd.Series:
    mapping = {"good": 0, "bad": 1}
    normalized = target.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized.unique()) - set(mapping))
    if unknown:
        raise ValueError(
            f"Unexpected target labels in OpenML credit-g dataset: {unknown}. "
            "Expected labels: ['good', 'bad']."
        )
    return normalized.map(mapping).astype(int)


def run_ingest() -> None:
    openml_cache = PROJECT_ROOT / ".cache" / "openml"
    sklearn_cache = PROJECT_ROOT / ".cache" / "sklearn"
    openml_cache.mkdir(parents=True, exist_ok=True)
    sklearn_cache.mkdir(parents=True, exist_ok=True)
    os.environ["OPENML_HOME"] = str(openml_cache)

    try:
        dataset = fetch_openml(name="credit-g", as_frame=True, data_home=str(sklearn_cache))
    except Exception as exc:
        raise RuntimeError(
            "Failed to download 'credit-g' from OpenML. "
            f"Check your internet connection and permissions. Original error: {exc}"
        ) from exc

    if dataset.data is None or dataset.target is None:
        raise RuntimeError("OpenML returned an empty dataset payload for 'credit-g'.")

    features_df = dataset.data.copy()
    target_series = _to_default_binary(dataset.target)
    df = features_df.copy()
    df["default"] = target_series

    RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(RAW_OUTPUT_PATH, index=False)

    target_rate = float(df["default"].mean())
    metadata = {
        "dataset_name": "credit-g",
        "source": "openml",
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "target_rate": target_rate,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    METADATA_OUTPUT_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"rows={metadata['rows']}")
    print(f"cols={metadata['cols']}")
    print(f"default_rate={target_rate:.4f}")
    print(f"output_path={RAW_OUTPUT_PATH}")


if __name__ == "__main__":
    run_ingest()
