# Reproducibility guide

## 1. Environment

The statistical analysis requires Python plus NumPy, pandas, SciPy, scikit-learn, and statsmodels. The supplied `requirements.txt` records the minimum versions used by the revision scripts.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Canonical input

All analyses consume:

```text
data/configurations.csv
```

The loader in `analysis/common.py` normalizes column names, constructs the canonical log variables, and creates the grouping identifiers used in train/test splitting.

## 3. Full reproduction

```bash
bash run_all.sh
```

Outputs are written below `outputs/reproduction/`.

## 4. Individual stages

```bash
python analysis/00_audit_dataset.py --input data/configurations.csv --outdir outputs/reproduction/00_audit
python analysis/01_grouped_validation.py --input data/configurations.csv --outdir outputs/reproduction/01_grouped_validation --settings bert gpt pooled --group-mode config_family --n-splits 20 --test-size 0.30 --seed 2026
python analysis/01_grouped_validation.py --input data/configurations.csv --outdir outputs/reproduction/01b_workload_family_validation --settings bert gpt pooled --group-mode workload_family --n-splits 20 --test-size 0.30 --seed 2026
python analysis/02_ablations_and_nonlinear_baselines.py --input data/configurations.csv --outdir outputs/reproduction/02_ablations --settings bert gpt pooled --group-mode config_family --n-splits 20 --test-size 0.30 --seed 2026
python analysis/03_nested_prerun_efficiency.py --input data/configurations.csv --outdir outputs/reproduction/03_prerun --settings bert gpt pooled --group-mode config_family --n-splits 20 --test-size 0.30 --seed 2026
python analysis/04_collinearity_proxy_inference.py --input data/configurations.csv --outdir outputs/reproduction/04_diagnostics --settings bert gpt pooled --group-mode config_family --n-splits 20 --test-size 0.30 --seed 2026
python analysis/05_hardware_and_matched_scaling.py --input data/configurations.csv --outdir outputs/reproduction/05_hardware_scaling --settings bert gpt pooled --n-splits 20 --test-size 0.30 --n-bootstrap 4000 --seed 2026
```

## 5. Measurement implementation

The CodeCarbon and NVML utilities are provided in `instrumentation/`. They document the measurement code but are not executed by the statistical reproduction pipeline.

## 6. Determinism

The outer repeated grouped splits use seed 2026. The nested pre-run predictor uses GroupKFold inside the outer training partition. Tree-based baselines derive deterministic seeds from the supplied base seed and split index.

Small floating-point differences may occur across BLAS/scikit-learn/statsmodels versions.
