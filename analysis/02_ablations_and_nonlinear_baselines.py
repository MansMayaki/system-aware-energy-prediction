#!/usr/bin/env python3
"""Systematic ablations and nonlinear baselines under identical grouped splits.

Reviewer coverage: R1 comments 6, 12, 13, 14, 19; establishes the incremental
value of compute, memory-related workload, and execution efficiency while
comparing the interpretable model with nonlinear regressors.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    add_group_ids, ensure_outdir, filter_hardware, group_column, load_data,
    onehot_encoder_dense, prediction_metrics, subset_setting,
    summarize_split_metrics, valid_group_splits,
)


def make_pipeline(numeric, categorical, model_kind="linear", seed=0):
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", onehot_encoder_dense()),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical),
    ], remainder="drop")

    if model_kind == "linear":
        reg = LinearRegression()
    elif model_kind == "rf":
        reg = RandomForestRegressor(
            n_estimators=600, min_samples_leaf=2, max_features="sqrt",
            random_state=seed, n_jobs=-1,
        )
    elif model_kind == "hgb":
        reg = HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
            l2_regularization=1e-3, random_state=seed,
        )
    else:
        raise ValueError(model_kind)
    return Pipeline([("pre", pre), ("reg", reg)])


def specs_for(df: pd.DataFrame, setting: str):
    cats_base = ["strategy"] + (["arch"] if setting == "pooled" else [])
    cats_hw = cats_base + (["gpu_model"] if "gpu_model" in df.columns else [])

    specs = {
        "tokens_only": (["log_tokens"], [], "linear"),
        "total_flops_only": (["log_total_flops"], [], "linear"),
        "compute_only": (["log_C"], [], "linear"),
        "compute_memory": (["log_C", "log_M"], [], "linear"),
        "compute_efficiency": (["log_C", "log_eta"], [], "linear"),
        "memory_efficiency": (["log_M", "log_eta"], [], "linear"),
        "full_interpretable": (["log_C", "log_M", "log_eta"], cats_base, "linear"),
        "full_plus_hardware": (["log_C", "log_M", "log_eta"], cats_hw, "linear"),
        "full_plus_hardware_gpu_count": (["log_C", "log_M", "log_eta", "log2_gpus"], cats_hw, "linear"),
        "runtime_only_postrun": (["log_duration"], [], "linear"),
        "device_seconds_postrun": (["log_device_seconds"], [], "linear"),
        # No target runtime/energy/eta; all inputs are known before execution.
        "configuration_only_prerun": (["log_C", "log_M", "gpus"], cats_hw, "linear"),
        # Nonlinear accuracy checks using exactly the full measured-eta predictor set.
        "random_forest_full_postrun": (["log_C", "log_M", "log_eta", "log2_gpus"], cats_hw, "rf"),
        "hist_gradient_boosting_full_postrun": (["log_C", "log_M", "log_eta", "log2_gpus"], cats_hw, "hgb"),
        # Nonlinear pre-run comparator: no runtime-derived quantities.
        "random_forest_configuration_prerun": (["log_C", "log_M", "gpus"], cats_hw, "rf"),
        "hist_gradient_boosting_configuration_prerun": (["log_C", "log_M", "gpus"], cats_hw, "hgb"),
    }
    # Drop specs whose required columns are unavailable.
    return {
        k: v for k, v in specs.items()
        if all(c in df.columns for c in v[0] + v[1])
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="outputs/revision/ablations")
    ap.add_argument("--settings", nargs="+", default=["bert", "gpt", "pooled"])
    ap.add_argument("--group-mode", default="config_family",
                    choices=["config_family", "workload_family"])
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--hardware", nargs="*", default=None)
    args = ap.parse_args()

    root = ensure_outdir(args.outdir)
    data = filter_hardware(add_group_ids(load_data(args.input)), args.hardware)
    gcol = group_column(args.group_mode)

    all_rows = []
    for setting in args.settings:
        d = subset_setting(data, setting).copy()
        d = d.dropna(subset=["log_energy", gcol]).copy()
        specs = specs_for(d, setting)
        cats_required = ["strategy"] + (["arch"] if setting == "pooled" else [])
        split_defs = list(valid_group_splits(
            d, gcol, n_splits=args.n_splits, test_size=args.test_size,
            seed=args.seed, categorical_cols=cats_required,
        ))

        print(f"\n[{setting}] rows={len(d)}, groups={d[gcol].nunique()}, specs={len(specs)}")
        for spec_name, (num, cat, kind) in specs.items():
            # Use exactly the same rows for every split/spec where its features exist.
            needed = ["log_energy", gcol] + num + cat
            rows = []
            for split_id, split_seed, tr_idx, te_idx in split_defs:
                train = d.iloc[tr_idx].dropna(subset=needed).copy()
                test = d.iloc[te_idx].dropna(subset=needed).copy()
                if len(train) < 20 or len(test) < 5:
                    continue
                pipe = make_pipeline(num, cat, model_kind=kind, seed=args.seed + split_id)
                pipe.fit(train[num + cat], train["log_energy"])
                pred = pipe.predict(test[num + cat])
                met = prediction_metrics(test["log_energy"], pred)
                met.update({
                    "setting": setting, "model": spec_name, "kind": kind,
                    "split": split_id, "split_seed": split_seed,
                    "n_train": len(train), "n_test": len(test),
                    "n_train_groups": train[gcol].nunique(),
                    "n_test_groups": test[gcol].nunique(),
                })
                rows.append(met)
            if not rows:
                continue
            res = pd.DataFrame(rows)
            all_rows.append(res)
            print(spec_name, "median R2=", round(res["r2"].median(), 4),
                  "RMSE=", round(res["rmse"].median(), 4))

    if not all_rows:
        raise RuntimeError("No ablation results were produced.")

    split_results = pd.concat(all_rows, ignore_index=True)
    split_results.to_csv(root / "all_split_results.csv", index=False)

    summaries = []
    for (setting, model), sub in split_results.groupby(["setting", "model"]):
        s = summarize_split_metrics(sub)
        s.insert(0, "model", model)
        s.insert(0, "setting", setting)
        summaries.append(s)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(root / "summary_long.csv", index=False)

    # Compact manuscript/rebuttal table: one row per setting/model.
    compact = []
    for (setting, model), sub in split_results.groupby(["setting", "model"]):
        compact.append({
            "setting": setting,
            "model": model,
            "R2_median": sub["r2"].median(),
            "R2_q25": sub["r2"].quantile(.25),
            "R2_q75": sub["r2"].quantile(.75),
            "RMSE_median": sub["rmse"].median(),
            "RMSE_q25": sub["rmse"].quantile(.25),
            "RMSE_q75": sub["rmse"].quantile(.75),
            "calib_slope_median": sub["calib_slope"].median(),
            "calib_intercept_median": sub["calib_intercept"].median(),
            "n_splits": len(sub),
        })
    pd.DataFrame(compact).sort_values(["setting", "R2_median"], ascending=[True, False]).to_csv(
        root / "summary_compact.csv", index=False
    )


if __name__ == "__main__":
    main()
