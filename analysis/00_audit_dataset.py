#!/usr/bin/env python3
"""Audit the final configuration-level dataset before any revision analysis.

Purpose
-------
Answers Reviewer 1 comments 1, 7, 16-18 and the corresponding questions about
repeated configurations and reproducibility. The script verifies that the model
input has one row per exact configuration, summarizes the number of raw repeated
runs represented by each row, and constructs the grouping variables used in all
subsequent train/test splits.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from common import (
    CONFIG_FAMILY_COLUMNS, WORKLOAD_FAMILY_COLUMNS, EXACT_CONFIG_COLUMNS,
    add_group_ids, ensure_outdir, load_data, save_json,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Final configuration-level CSV")
    ap.add_argument("--outdir", default="outputs/revision/audit")
    ap.add_argument("--expected-configs", type=int, default=None)
    ap.add_argument("--expected-runs", type=int, default=None)
    ap.add_argument("--strict", action="store_true",
                    help="Fail if expected counts do not match")
    args = ap.parse_args()

    out = ensure_outdir(args.outdir)
    df = add_group_ids(load_data(args.input))

    n_rows = len(df)
    exact_dups = int(df["exact_config_id"].duplicated().sum())
    n_exact = int(df["exact_config_id"].nunique())
    n_config_families = int(df["config_family_id"].nunique())
    n_workload_families = int(df["workload_family_id"].nunique())

    if "n_runs" in df.columns:
        n_runs_sum = float(pd.to_numeric(df["n_runs"], errors="coerce").sum())
        n_runs_desc = pd.to_numeric(df["n_runs"], errors="coerce").describe().to_dict()
    else:
        n_runs_sum = None
        n_runs_desc = None

    summary = {
        "input": str(Path(args.input).resolve()),
        "configuration_rows": n_rows,
        "unique_exact_configurations": n_exact,
        "duplicate_exact_configuration_rows": exact_dups,
        "configuration_families": n_config_families,
        "workload_families_strict": n_workload_families,
        "sum_n_runs": n_runs_sum,
        "n_runs_distribution": n_runs_desc,
        "config_family_columns_used": [c for c in CONFIG_FAMILY_COLUMNS if c in df.columns],
        "workload_family_columns_used": [c for c in WORKLOAD_FAMILY_COLUMNS if c in df.columns],
        "exact_config_columns_used": [c for c in EXACT_CONFIG_COLUMNS if c in df.columns],
    }

    warnings = []
    if exact_dups:
        warnings.append(
            f"Found {exact_dups} duplicate exact-configuration rows. Aggregate repetitions "
            "before modeling."
        )
    if args.expected_configs is not None and n_rows != args.expected_configs:
        warnings.append(
            f"Expected {args.expected_configs} configuration rows but found {n_rows}."
        )
    if args.expected_runs is not None:
        if n_runs_sum is None:
            warnings.append("Expected raw-run count requested but n_runs is absent.")
        elif int(round(n_runs_sum)) != args.expected_runs:
            warnings.append(
                f"Expected {args.expected_runs} raw runs but sum(n_runs)={n_runs_sum:g}."
            )
    summary["warnings"] = warnings

    save_json(summary, out / "dataset_audit.json")

    # Counts used directly in the rebuttal/manuscript.
    counts = []
    for keys in [
        ["arch"], ["gpu_model"], ["strategy"], ["arch", "strategy"],
        ["gpu_model", "strategy"], ["arch", "gpu_model", "strategy"],
        ["gpus"], ["arch", "gpus"],
    ]:
        keys = [k for k in keys if k in df.columns]
        if not keys:
            continue
        tmp = df.groupby(keys, dropna=False).size().rename("n_configurations").reset_index()
        tmp["breakdown"] = " x ".join(keys)
        counts.append(tmp)
    if counts:
        pd.concat(counts, ignore_index=True, sort=False).to_csv(out / "dataset_counts.csv", index=False)

    # Distribution of repeated raw executions per configuration.
    if "n_runs" in df.columns:
        (
            df.groupby("n_runs", dropna=False).size().rename("n_configurations").reset_index()
            .sort_values("n_runs")
            .to_csv(out / "repetitions_per_configuration.csv", index=False)
        )

    # Save an analysis-ready normalized table with explicit group IDs.
    df.to_csv(out / "analysis_ready_with_group_ids.csv", index=False)

    print("=== DATASET AUDIT ===")
    for k, v in summary.items():
        if k not in {"n_runs_distribution", "warnings"}:
            print(f"{k}: {v}")
    if n_runs_desc is not None:
        print("n_runs distribution:", n_runs_desc)
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(" -", w)
    else:
        print("\nNo audit warnings.")

    if args.strict and warnings:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
