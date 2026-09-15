#!/usr/bin/env bash
set -euo pipefail

INPUT=${1:-data/configurations.csv}
OUT=${2:-outputs/reproduction}

python analysis/00_audit_dataset.py \
  --input "$INPUT" \
  --outdir "$OUT/00_audit"

python analysis/01_grouped_validation.py \
  --input "$INPUT" \
  --outdir "$OUT/01_grouped_validation" \
  --settings bert gpt pooled \
  --group-mode config_family \
  --n-splits 20 \
  --test-size 0.30 \
  --seed 2026

python analysis/01_grouped_validation.py \
  --input "$INPUT" \
  --outdir "$OUT/01b_workload_family_validation" \
  --settings bert gpt pooled \
  --group-mode workload_family \
  --n-splits 20 \
  --test-size 0.30 \
  --seed 2026

python analysis/02_ablations_and_nonlinear_baselines.py \
  --input "$INPUT" \
  --outdir "$OUT/02_ablations" \
  --settings bert gpt pooled \
  --group-mode config_family \
  --n-splits 20 \
  --test-size 0.30 \
  --seed 2026

python analysis/03_nested_prerun_efficiency.py \
  --input "$INPUT" \
  --outdir "$OUT/03_prerun" \
  --settings bert gpt pooled \
  --group-mode config_family \
  --n-splits 20 \
  --test-size 0.30 \
  --seed 2026

python analysis/04_collinearity_proxy_inference.py \
  --input "$INPUT" \
  --outdir "$OUT/04_diagnostics" \
  --settings bert gpt pooled \
  --group-mode config_family \
  --n-splits 20 \
  --test-size 0.30 \
  --seed 2026

python analysis/05_hardware_and_matched_scaling.py \
  --input "$INPUT" \
  --outdir "$OUT/05_hardware_scaling" \
  --settings bert gpt pooled \
  --n-splits 20 \
  --test-size 0.30 \
  --n-bootstrap 4000 \
  --seed 2026

echo "Reproduction analyses completed: $OUT"
