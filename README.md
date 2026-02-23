# mini-mrm-credit-validation-pack

Minimal scaffold for a Model Risk Management (MRM) credit validation workflow. 
It includes folders for raw/processed data, SQL transforms, source modules, report figures, and output artifacts.

## Purpose
This project is intended as a starter template to:
- ingest and quality-check credit data,
- engineer features,
- train baseline models,
- run validation checks,
- store outputs and metrics in a reproducible structure.

## How to run
1. Create and activate the virtual environment:
   - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Start implementing pipeline modules in `src/`:
   - `ingest.py`, `qc.py`, `features.py`, `train.py`, `validate.py`

You can add CLI entry points or notebooks as needed.

## Quickstart
```bash
source .venv/bin/activate
python -m src.ingest
python -m src.build_table
python -m src.qc
python -m src.train
python -m src.validate
```

Expected output (example):
```text
rows=1000
columns=21
target_rate=0.3000
output_file=/.../mini-mrm-credit-validation-pack/data/raw/credit_g.csv
```

This command writes:
- raw data to `data/raw/credit_g.csv`
- metadata log to `outputs/logs/ingest_metadata.json`

Build table command outputs:
- processed modeling table: `data/processed/model_table.csv`
- build metadata log: `outputs/logs/model_table_metadata.json`

QC command outputs:
- `outputs/metrics/qc_missingness.csv`
- `outputs/metrics/qc_profile_numeric.csv`
- `outputs/metrics/qc_profile_categorical.csv`
- `outputs/logs/qc_report.json`

Train command outputs:
- model pipeline: `outputs/models/logit_pd_pipeline.joblib`
- training metadata: `outputs/logs/train_metadata.json`

Validate command outputs:
- metrics: `outputs/metrics/validation_metrics.json`
- PSI summary: `outputs/metrics/psi_score.json`
- figures:
  - `reports/figures/roc_curve_test.png`
  - `reports/figures/calibration_curve_test.png`
  - `reports/figures/score_hist_test.png`

## Run End-To-End
```bash
source .venv/bin/activate
python -m src.run_all
```

This run writes:
- `data/raw/credit_g.csv`
- `data/processed/model_table.csv`
- `outputs/models/logit_pd_pipeline.joblib`
- `outputs/metrics/validation_metrics.json`
- `reports/figures/roc_curve_test.png`
- `reports/figures/calibration_curve_test.png`
- `reports/figures/score_hist_test.png`
- `reports/validation_memo.md`
