from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "credit_g.csv"
SQL_PATH = PROJECT_ROOT / "sql" / "01_build_model_table.sql"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "model_table.csv"
METADATA_PATH = PROJECT_ROOT / "outputs" / "logs" / "model_table_metadata.json"


def run_build_table() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}. Run `python -m src.ingest` first."
        )
    if not SQL_PATH.exists():
        raise FileNotFoundError(f"Missing SQL file: {SQL_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    query = SQL_PATH.read_text(encoding="utf-8")
    with duckdb.connect(database=":memory:") as conn:
        conn.execute(f"SET file_search_path='{PROJECT_ROOT.as_posix()}'")
        model_df = conn.execute(query).fetchdf()

    model_df.to_csv(OUTPUT_PATH, index=False)

    rows, cols = model_df.shape
    default_rate = float(model_df["default"].mean())
    numeric_columns = list(model_df.select_dtypes(include="number").columns)
    categorical_columns = list(model_df.select_dtypes(exclude="number").columns)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "rows": int(rows),
        "cols": int(cols),
        "default_rate": default_rate,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"rows={rows}")
    print(f"cols={cols}")
    print(f"default_rate={default_rate:.4f}")
    print(f"output_path={OUTPUT_PATH}")


if __name__ == "__main__":
    run_build_table()
