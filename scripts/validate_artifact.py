#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
required = [
    ROOT / "README.md",
    ROOT / "requirements.txt",
    ROOT / "run_all.sh",
    ROOT / "analysis/common.py",
    ROOT / "data/configurations.csv",
    ROOT / "data/data_dictionary.csv",
    ROOT / "instrumentation/codecarbon_tracker.py",
    ROOT / "instrumentation/nvml_sampler.py",
]
missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
if missing:
    raise SystemExit("Missing required artifact files: " + ", ".join(missing))

df = pd.read_csv(ROOT / "data/configurations.csv")
for c in ["arch", "strategy", "gpus", "energy_consumed_kWh", "config_family_id", "workload_family_id"]:
    if c not in df.columns:
        raise SystemExit(f"Missing required data column: {c}")
if (pd.to_numeric(df["energy_consumed_kWh"], errors="coerce") <= 0).any():
    raise SystemExit("Non-positive energy target found.")

summary = {
    "rows": int(len(df)),
    "config_families": int(df["config_family_id"].nunique()),
    "workload_families": int(df["workload_family_id"].nunique()),
    "gpu_models": sorted(df["gpu_model"].dropna().astype(str).unique().tolist()),
    "strategies": sorted(df["strategy"].dropna().astype(str).unique().tolist()),
    "power_correction_values": sorted(pd.to_numeric(df["power_correction"], errors="coerce").dropna().unique().tolist()) if "power_correction" in df else [],
}
print(json.dumps(summary, indent=2))
print("Artifact structure validation passed.")
