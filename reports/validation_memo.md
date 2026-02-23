# Validation Memo: Baseline PD Model (German Credit)

## Conclusion
- Overall outcome: the baseline PD model is acceptable for demo validation and baseline benchmarking purposes.
- Strengths: discrimination is solid (AUC test 0.818, KS test 0.590), calibration error is low/stable (Brier test 0.147), and score stability is strong (PSI 0.018).
- Key risks: this uses a public benchmark dataset with a proxy default label and a random split design.
- Drift caveat: PSI 0.018 indicates minimal shift under a random holdout split; this is not a time-based drift test.
- Production preconditions: require time-based out-of-sample testing, formal monitoring thresholds, and governance sign-off before any production use.
- Validation conclusion: use as controlled demo/baseline only, not decisioning production.

## Purpose & Scope
This memo summarizes independent-style validation checks for a baseline probability of default (PD) model built on the German credit (credit-g) dataset. The objective is to provide a compact demonstration Model Risk Management (MRM) validation pack for a simple logistic regression scorecard-style model.

## Model Use
- Intended use: educational/demo baseline for binary default risk ranking and relative PD estimation.
- Out of scope: production underwriting decisions, pricing, capital, provisioning, or regulatory reporting.
- Key assumption: the `default` label in the dataset is treated as the observed event indicator (1=bad credit, 0=good credit).

## Data Description
- Source dataset: OpenML `credit-g`.
- Observations: 1,000 rows.
- Target: `default` with event rate of 30%.
- Data flow: raw extract (`data/raw/credit_g.csv`) -> processed modeling table (`data/processed/model_table.csv`).
- Feature engineering in processed table includes:
  - `credit_amount_log = ln(credit_amount + 1)`
  - `credit_per_month = credit_amount / duration` (with divide-by-zero protection)
- Completeness checks showed 0% missingness across fields; see `outputs/metrics/qc_missingness.csv` and QC profile outputs.

## Methodology
- Split: holdout train/test split configured in `config.yaml` (`random_seed=42`, `test_size=0.2`), stratified on target.
- Preprocessing: column-wise transformation using one-hot encoding for categorical variables (`handle_unknown='ignore'`), numeric passthrough.
- Model: logistic regression (`solver='liblinear'`, `max_iter=200`) in an sklearn pipeline.
- Validation metrics produced on both train and test where applicable:
  - Discrimination: AUC, KS
  - Calibration accuracy: Brier score
  - Stability: PSI on model score (train vs test), bins from `psi_bins` in config

## Results
| Metric | Value |
|---|---:|
| AUC (train) | 0.833 |
| AUC (test) | 0.818 |
| KS (train) | 0.511 |
| KS (test) | 0.590 |
| Brier (train) | 0.147 |
| Brier (test) | 0.147 |
| PSI score (train vs test) | 0.018 |

## Interpretation
- Discrimination: test AUC and KS indicate meaningful separation between default and non-default observations.
- Calibration: train and test Brier scores are close, suggesting similar probability error magnitude across splits.
- Stability: PSI is low, indicating limited distributional shift in model scores between train and test.
- MRM view: validation outcomes are acceptable for baseline/demo use, with model risk controls required before broader deployment.

## Limitations & Model Risk Items
- Public benchmark dataset; limited representativeness for institution-specific portfolios.
- Target is a proxy label from the source dataset, not an internal performance definition.
- Random split only; no time-based out-of-sample validation.
- No fairness/bias testing across protected or sensitive segments.
- No reject inference or selection-bias correction.
- Limited feature governance/lineage and no challenger model benchmark.
- No stress/scenario or macro-sensitivity testing.

## Model Governance
- Versioned artifacts:
  - model: `outputs/models/logit_pd_pipeline.joblib`
  - training metadata: `outputs/logs/train_metadata.json`
  - validation metrics: `outputs/metrics/validation_metrics.json`
- Validation timestamp (UTC): 2026-02-23T22:51:00.915626+00:00
- Governance expectation: retain reproducible run configs, lock data snapshots, and track preprocessing/model changes across versions.

## Next Steps
1. Implement monitoring cadence (monthly): PSI, AUC/KS drift, and calibration drift tracking.
2. Add time-based validation split once temporal data is available.
3. Expand validation scope with fairness testing, segmentation analysis, and challenger models.
4. Add fuller documentation (data dictionary, assumptions log, validation checklist, and sign-off workflow).
