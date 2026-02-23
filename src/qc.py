from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "credit_g.csv"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml must contain a top-level mapping.")
    return data


def _build_missingness(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, int]:
    missing_count = df.isna().sum()
    missing_rate = missing_count / len(df)
    result = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": missing_count.values.astype(int),
            "missing_rate": missing_rate.values.astype(float),
        }
    )
    result["status"] = result["missing_rate"].apply(
        lambda x: "FAIL" if x > threshold else "PASS"
    )
    n_failed = int((result["status"] == "FAIL").sum())
    return result, n_failed


def _build_numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = list(df.select_dtypes(include="number").columns)
    rows: list[dict[str, Any]] = []
    for col in numeric_cols:
        series = df[col]
        rows.append(
            {
                "column": col,
                "min": series.min(),
                "p1": series.quantile(0.01),
                "p50": series.quantile(0.50),
                "p99": series.quantile(0.99),
                "max": series.max(),
            }
        )
    return pd.DataFrame(rows)


def _build_categorical_profile(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = list(df.select_dtypes(exclude="number").columns)
    rows: list[dict[str, Any]] = []
    for col in categorical_cols:
        series = df[col]
        n_unique = int(series.nunique(dropna=True))
        value_counts = series.astype(str).value_counts(dropna=False).head(5)
        total = len(series)
        for rank, (category, count) in enumerate(value_counts.items(), start=1):
            rows.append(
                {
                    "column": col,
                    "n_unique": n_unique,
                    "rank": rank,
                    "category": category,
                    "count": int(count),
                    "rate": float(count / total),
                }
            )
    return pd.DataFrame(rows)


def run_qc() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}. Run `python -m src.ingest` first."
        )

    config = _load_config(CONFIG_PATH)
    threshold = float(config.get("missingness_fail_threshold", 0.01))

    df = pd.read_csv(INPUT_PATH)
    if "default" not in df.columns:
        raise ValueError("Expected target column `default` in data/raw/credit_g.csv.")

    unique_target = sorted(df["default"].dropna().unique().tolist())
    if not set(unique_target).issubset({0, 1}):
        raise ValueError(
            f"Target column `default` must be binary (0/1), found values: {unique_target}"
        )

    rows, cols = df.shape
    default_rate = float(df["default"].mean())
    n_duplicates = int(df.duplicated().sum())

    missingness_df, n_failed_missingness = _build_missingness(df, threshold)
    numeric_profile_df = _build_numeric_profile(df)
    categorical_profile_df = _build_categorical_profile(df)

    overall_status = "PASS"
    if n_failed_missingness > 0 or n_duplicates > 0:
        overall_status = "FAIL"

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    missingness_df.to_csv(METRICS_DIR / "qc_missingness.csv", index=False)
    numeric_profile_df.to_csv(METRICS_DIR / "qc_profile_numeric.csv", index=False)
    categorical_profile_df.to_csv(METRICS_DIR / "qc_profile_categorical.csv", index=False)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(rows),
        "cols": int(cols),
        "default_rate": default_rate,
        "n_duplicates": n_duplicates,
        "n_columns_failed_missingness": n_failed_missingness,
        "overall_status": overall_status,
    }
    (LOGS_DIR / "qc_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"overall_status={overall_status}")
    print(f"rows={rows}, cols={cols}")
    print(f"default_unique={unique_target}, default_rate={default_rate:.4f}")
    print(f"n_duplicates={n_duplicates}")
    print(f"n_columns_failed_missingness={n_failed_missingness} (threshold={threshold})")
    print(f"metrics_dir={METRICS_DIR}")
    print(f"report_path={LOGS_DIR / 'qc_report.json'}")


if __name__ == "__main__":
    run_qc()
