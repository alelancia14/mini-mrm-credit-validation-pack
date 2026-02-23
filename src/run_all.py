from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    try:
        import certifi

        env["SSL_CERT_FILE"] = certifi.where()
    except Exception:
        pass
    return env


def _run_step(module_name: str, env: dict[str, str]) -> None:
    print(f"START {module_name}")
    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        print(f"FAIL  {module_name}")
        raise SystemExit(result.returncode)
    print(f"OK    {module_name}")


def run_all() -> None:
    env = _base_env()
    steps = [
        "src.ingest",
        "src.qc",
        "src.build_table",
        "src.train",
        "src.validate",
    ]
    for step in steps:
        _run_step(step, env)

    print("Artifacts:")
    print(f"- {PROJECT_ROOT / 'data/raw/credit_g.csv'}")
    print(f"- {PROJECT_ROOT / 'data/processed/model_table.csv'}")
    print(f"- {PROJECT_ROOT / 'outputs/models/logit_pd_pipeline.joblib'}")
    print(f"- {PROJECT_ROOT / 'outputs/metrics/validation_metrics.json'}")
    print(f"- {PROJECT_ROOT / 'reports/figures/*.png'}")
    print(f"- {PROJECT_ROOT / 'reports/validation_memo.md'}")


if __name__ == "__main__":
    run_all()
