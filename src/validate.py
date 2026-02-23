from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "model_table.csv"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "logit_pd_pipeline.joblib"
METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "validation_metrics.json"
PSI_PATH = PROJECT_ROOT / "outputs" / "metrics" / "psi_score.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("config.yaml must contain a top-level mapping.")
    return raw


def _ks_stat(y_true: pd.Series, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int) -> float:
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.unique(np.quantile(expected, quantiles))
    if len(bins) < 2:
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)
    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)

    eps = 1e-6
    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _plot_roc_test(y_test: pd.Series, p_test: np.ndarray, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_test, p_test)
    auc_test = roc_auc_score(y_test, p_test)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_test:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def _plot_calibration_test(
    y_test: pd.Series, p_test: np.ndarray, output_path: Path
) -> None:
    prob_true, prob_pred = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Default Rate")
    plt.title("Calibration Curve (Test)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def _plot_score_hist_test(
    y_test: pd.Series, p_test: np.ndarray, output_path: Path
) -> None:
    plt.figure(figsize=(6, 5))
    plt.hist(
        p_test[y_test == 0],
        bins=20,
        alpha=0.6,
        label="default=0",
        density=True,
    )
    plt.hist(
        p_test[y_test == 1],
        bins=20,
        alpha=0.6,
        label="default=1",
        density=True,
    )
    plt.xlabel("Predicted PD")
    plt.ylabel("Density")
    plt.title("Score Distribution (Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def run_validate() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}. Run `python -m src.build_table` first."
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}. Run `python -m src.train` first."
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

    model = joblib.load(MODEL_PATH)
    p_train = model.predict_proba(X_train)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    auc_train = float(roc_auc_score(y_train, p_train))
    auc_test = float(roc_auc_score(y_test, p_test))
    ks_train = _ks_stat(y_train, p_train)
    ks_test = _ks_stat(y_test, p_test)
    brier_train = float(brier_score_loss(y_train, p_train))
    brier_test = float(brier_score_loss(y_test, p_test))
    psi_score = _psi(p_train, p_test, psi_bins)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    _plot_roc_test(y_test, p_test, FIGURES_DIR / "roc_curve_test.png")
    _plot_calibration_test(y_test, p_test, FIGURES_DIR / "calibration_curve_test.png")
    _plot_score_hist_test(y_test, p_test, FIGURES_DIR / "score_hist_test.png")

    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "auc_train": auc_train,
        "auc_test": auc_test,
        "ks_train": ks_train,
        "ks_test": ks_test,
        "brier_train": brier_train,
        "brier_test": brier_test,
        "psi_score_train_vs_test": psi_score,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    PSI_PATH.write_text(
        json.dumps({"psi_score_train_vs_test": psi_score}, indent=2), encoding="utf-8"
    )

    print(f"auc_train={auc_train:.4f}, auc_test={auc_test:.4f}")
    print(f"ks_train={ks_train:.4f}, ks_test={ks_test:.4f}")
    print(f"brier_train={brier_train:.4f}, brier_test={brier_test:.4f}")
    print(f"psi_score_train_vs_test={psi_score:.4f} (bins={psi_bins})")
    print(f"metrics_path={METRICS_PATH}")
    print(f"figures_dir={FIGURES_DIR}")


if __name__ == "__main__":
    run_validate()
