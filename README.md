# System-Aware Energy Prediction for Multi-GPU Transformer Training

Anonymous reproducibility artifact accompanying the manuscript **“System-Aware Energy Prediction for Multi-GPU Transformer Training.”**

This repository contains the analysis-ready configuration-level dataset, statistical analysis code, energy-instrumentation utilities, and reference outputs used for the revised evaluation. Author names, affiliations, personal filesystem paths, and Git history are intentionally absent.

## Scope

The study evaluates telemetry-derived operational training energy for dense Transformer workloads under single-node NVIDIA multi-GPU execution. The released analysis covers BERT- and GPT-family workloads, Baseline/FSDP execution, TP where available, and A100, RTX 2080 Ti, T4, and V100 observations.

The principal reduced-form model is

```text
log E = beta_0 + alpha_C log C + alpha_M log M_proxy + alpha_eta log eta_h + execution-regime terms + error.
```

The coefficients are interpreted as conditional empirical elasticities within the observed regimes, not as universal physical constants or causal effects.

## Repository layout

```text
.
├── README.md
├── REPRODUCIBILITY.md
├── requirements.txt
├── environment.yml
├── run_all.sh
├── analysis/
│   ├── common.py
│   ├── 00_audit_dataset.py
│   ├── 01_grouped_validation.py
│   ├── 02_ablations_and_nonlinear_baselines.py
│   ├── 03_nested_prerun_efficiency.py
│   ├── 04_collinearity_proxy_inference.py
│   └── 05_hardware_and_matched_scaling.py
├── data/
│   ├── configurations.csv
│   ├── data_dictionary.csv
│   └── README.md
├── instrumentation/
│   ├── codecarbon_tracker.py
│   ├── nvml_sampler.py
│   ├── tp_trainer_reference.py
│   └── README.md
├── configs/
│   ├── evaluation.yaml
│   └── seeds.txt
├── results/
│   ├── README.md
│   └── reference_outputs/
└── scripts/
    ├── check_anonymity.sh
    └── validate_artifact.py
```

## Released analysis table

`data/configurations.csv` is the canonical input for all analysis scripts in this artifact. The supplied release contains **693 configuration-level rows**, **246 configuration families**, and **135 stricter workload families**. Repeated source executions are already aggregated; `n_runs` records how many retained source executions are represented by each row.

The full raw epoch-level training archive is not required to reproduce the statistical analyses and is not included in this anonymous artifact.

See `data/data_dictionary.csv` for definitions and units.

## Energy measurement

The training code uses the same CodeCarbon measurement wrapper across hardware platforms. For each measured training epoch, an `EmissionsTracker` is started immediately before the training loop and stopped immediately after it. The source training code uses:

```python
measure_power_secs = 0.1
tracking_mode = "process"
```

The primary energy variable in the released table is `energy_consumed_kWh`. In the supplied analysis table, `power_correction` is 1.0 for all rows, so no hardware-dependent rescaling is present in the released data.

An independent NVIDIA NVML sampler additionally records aggregate visible-GPU power at 0.1-s intervals and integrates the power trace with the trapezoidal rule. This auxiliary measurement documents the device-level telemetry path but is not used to redefine the primary statistical target.

The artifact therefore treats the target as **telemetry-derived operational training energy under a common measurement protocol**, not as wall-plug, facility, lifecycle, or embodied energy.

## Validation design

The headline evaluation is configuration-family-disjoint. The group key fixes the workload/hardware descriptors but excludes GPU count and execution strategy, so Baseline/FSDP/TP and GPU-count variants of a workload cannot be split between train and test.

A stricter workload-family analysis also excludes hardware from the grouping key, preventing the same architectural workload from appearing on different GPU families across train and test.

The supplied scripts use 20 repeated grouped 70/30 splits with seed 2026. The prospective efficiency analysis uses five-fold grouped cross-fitting entirely inside each outer training partition.

## Quick start

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then reproduce all statistical analyses:

```bash
bash run_all.sh
```

Results are written to `outputs/reproduction/`.

To verify only the artifact structure and anonymity checks:

```bash
python scripts/validate_artifact.py
bash scripts/check_anonymity.sh
```

## Reference outputs

`results/reference_outputs/` contains the outputs from the supplied revision-analysis run, including grouped validation, stricter workload-family validation, ablations, pre-run efficiency prediction, collinearity/proxy diagnostics, hardware sensitivity, leave-one-hardware-out transfer, and matched scaling.

Representative median configuration-family-disjoint held-out R2 values in the included reference outputs are approximately 0.937 (BERT), 0.927 (GPT), and 0.922 (pooled). Under the stricter workload-family split they are approximately 0.925, 0.926, and 0.927, respectively.

## Important interpretation notes

- `M_proxy` is a memory-related workload descriptor, not measured HBM traffic.
- Measured `eta_h` is post-run. The nested pre-run analysis predicts efficiency from configuration information using grouped cross-fitting.
- Runtime, device-seconds, measured efficiency, and telemetry-based quantities are post-run variables and are not used as target-run inputs to the prospective efficiency predictor.
- Cross-hardware experiments are transfer stress tests; the fitted coefficients are expected to require recalibration outside the observed hardware/workload envelope.
- The validated empirical scope is single-node dense-Transformer training. The artifact does not establish multi-node, Mixture-of-Experts, inference, or lifecycle-energy results.

## Anonymity

This peer-review artifact intentionally omits author identity, affiliation, acknowledgments, personal cluster usernames, Git history, and non-anonymous repository URLs.

## Review-use notice

This anonymous copy is supplied for scholarly peer review and reproducibility inspection. A final attributed archive and permanent citation can be created after the review process.
