#!/usr/bin/env python3
"""Primary configuration-disjoint validation for BERT, GPT, and pooled models.

This script is designed to replace the old random row-wise 70/30 split as the
headline validation. Related multi-GPU variants of the same workload are kept in
one partition by grouping on configuration family (strategy and GPU count are
excluded from the group key).

Outputs per setting:
  - split_metrics.csv
  - metric_summary.csv
  - cluster_robust_coefficients.csv
  - conformal_interval_metrics.csv
  - conformal_interval_summary.csv

Reviewer coverage: R1 comments 1, 7, 12, 15 and questions 1, 2, 9, 11.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from common import (
    add_group_ids, conformal_q, ensure_outdir, filter_hardware, full_formula,
    group_column, load_data, prediction_metrics, subset_setting,
    summarize_split_metrics, valid_group_splits,
)

CENTER = ["log_C", "log_M", "log_eta"]


def clean_for_model(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["log_energy", "log_C", "log_M", "log_eta", "strategy", "gpus"]
    d = df.dropna(subset=[c for c in needed if c in df.columns]).copy()
    d = d[np.isfinite(d["log_energy"]) & np.isfinite(d["log_C"]) &
          np.isfinite(d["log_M"]) & np.isfinite(d["log_eta"])].copy()
    return d


def center(train: pd.DataFrame, other: pd.DataFrame):
    train = train.copy(); other = other.copy()
    for c in CENTER:
        mu = float(train[c].mean())
        train[c + "_c"] = train[c] - mu
        other[c + "_c"] = other[c] - mu
    return train, other


def fit_formula(setting: str, train: pd.DataFrame, test: pd.DataFrame):
    tr, te = center(train, test)
    f = full_formula(setting, eta_col="log_eta", include_hardware=False, interactions=False)
    m = smf.ols(f, data=tr).fit(cov_type="HC3")
    pred = np.asarray(m.predict(te), dtype=float)
    return m, pred


def evaluate_grouped(df: pd.DataFrame, setting: str, group_col: str,
                     n_splits: int, test_size: float, seed: int):
    rows = []
    categorical = ["strategy"] + (["arch"] if setting == "pooled" else [])
    for split_id, split_seed, tr_idx, te_idx in valid_group_splits(
        df, group_col, n_splits=n_splits, test_size=test_size,
        seed=seed, categorical_cols=categorical,
    ):
        train, test = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        model, pred = fit_formula(setting, train, test)
        met = prediction_metrics(test["log_energy"], pred)
        met.update({
            "split": split_id,
            "split_seed": split_seed,
            "n_train": len(train),
            "n_train_groups": train[group_col].nunique(),
            "n_test_groups": test[group_col].nunique(),
        })
        rows.append(met)
    return pd.DataFrame(rows)



def evaluate_rowwise(df: pd.DataFrame, setting: str, n_splits: int, test_size: float, seed: int):
    """Historical row-wise split, reported only as a sensitivity comparison."""
    rows = []
    rng = np.random.default_rng(seed)
    for split_id in range(n_splits):
        rs = int(rng.integers(0, 2**31 - 1))
        # Joint stratification can fail for rare cells; fall back to strategy.
        strat = None
        if "strategy" in df.columns:
            joint = df["strategy"].astype(str) + "|" + df["gpus"].astype(str)
            if joint.value_counts().min() >= 2:
                strat = joint
            elif df["strategy"].value_counts().min() >= 2:
                strat = df["strategy"]
        tr, te = train_test_split(df, test_size=test_size, random_state=rs, stratify=strat)
        _, pred = fit_formula(setting, tr, te)
        met = prediction_metrics(te["log_energy"], pred)
        met.update({"split": split_id, "split_seed": rs, "n_train": len(tr), "n_test": len(te)})
        rows.append(met)
    return pd.DataFrame(rows)

def split_conformal_once(df: pd.DataFrame, setting: str, group_col: str,
                         outer_train_idx, test_idx, seed: int, calib_frac: float = 0.20,
                         alpha: float = 0.05):
    outer_train = df.iloc[outer_train_idx].copy()
    test = df.iloc[test_idx].copy()

    gss = GroupShuffleSplit(n_splits=1, test_size=calib_frac, random_state=seed)
    fit_rel, cal_rel = next(gss.split(outer_train, groups=outer_train[group_col]))
    fit = outer_train.iloc[fit_rel].copy()
    cal = outer_train.iloc[cal_rel].copy()

    # Ensure all categorical levels in calibration/test are known to fit.
    for c in ["strategy"] + (["arch"] if setting == "pooled" else []):
        if not set(cal[c]).issubset(set(fit[c])) or not set(test[c]).issubset(set(fit[c])):
            return None

    fit_c, cal_c = center(fit, cal)
    _, test_c = center(fit, test)
    f = full_formula(setting, eta_col="log_eta", include_hardware=False, interactions=False)
    model = smf.ols(f, data=fit_c).fit()
    pred_cal = np.asarray(model.predict(cal_c), dtype=float)
    pred_test = np.asarray(model.predict(test_c), dtype=float)

    q = conformal_q(np.abs(cal["log_energy"].to_numpy() - pred_cal), alpha=alpha)
    y = test["log_energy"].to_numpy(dtype=float)
    lo, hi = pred_test - q, pred_test + q
    covered = (y >= lo) & (y <= hi)

    # Log interval width and a more interpretable multiplicative factor.
    return {
        "coverage": float(np.mean(covered)),
        "q_log": float(q),
        "log_interval_width": float(2 * q),
        "multiplicative_half_width": float(np.exp(q)),
        "median_interval_width_kwh": float(np.median(np.exp(hi) - np.exp(lo))),
        "n_fit": len(fit), "n_cal": len(cal), "n_test": len(test),
        "n_fit_groups": fit[group_col].nunique(),
        "n_cal_groups": cal[group_col].nunique(),
        "n_test_groups": test[group_col].nunique(),
    }


def evaluate_conformal(df: pd.DataFrame, setting: str, group_col: str,
                       n_splits: int, test_size: float, seed: int):
    rows = []
    categorical = ["strategy"] + (["arch"] if setting == "pooled" else [])
    for split_id, split_seed, tr_idx, te_idx in valid_group_splits(
        df, group_col, n_splits=n_splits, test_size=test_size,
        seed=seed + 991, categorical_cols=categorical,
    ):
        result = None
        # Retry the internal fit/calibration grouping if a rare category lands
        # entirely in calibration.
        for j in range(50):
            result = split_conformal_once(
                df, setting, group_col, tr_idx, te_idx,
                seed=split_seed + j, calib_frac=0.20, alpha=0.05,
            )
            if result is not None:
                break
        if result is None:
            continue
        result.update({"split": split_id, "split_seed": split_seed})
        rows.append(result)
    return pd.DataFrame(rows)


def cluster_robust_full_fit(df: pd.DataFrame, setting: str, group_col: str):
    d = df.copy()
    for c in CENTER:
        d[c + "_c"] = d[c] - d[c].mean()
    f = full_formula(setting, eta_col="log_eta", include_hardware=False, interactions=False)
    m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d[group_col]})
    ci = m.conf_int(alpha=0.05)
    return pd.DataFrame({
        "term": m.params.index,
        "coef": m.params.values,
        "cluster_se": m.bse.values,
        "z": m.tvalues.values,
        "pvalue": m.pvalues.values,
        "ci_low": ci.iloc[:, 0].values,
        "ci_high": ci.iloc[:, 1].values,
        "n": len(d),
        "n_clusters": d[group_col].nunique(),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="outputs/revision/grouped_validation")
    ap.add_argument("--settings", nargs="+", default=["bert", "gpt", "pooled"])
    ap.add_argument("--group-mode", default="config_family",
                    choices=["config_family", "workload_family"])
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--hardware", nargs="*", default=None,
                    help="Optional exact gpu_model values to retain")
    args = ap.parse_args()

    root = ensure_outdir(args.outdir)
    data = add_group_ids(load_data(args.input))
    data = filter_hardware(data, args.hardware)
    gcol = group_column(args.group_mode)

    all_summary = []
    for setting in args.settings:
        out = ensure_outdir(root / setting)
        d = clean_for_model(subset_setting(data, setting))
        print(f"\n[{setting}] rows={len(d)}, {gcol}={d[gcol].nunique()}")
        if len(d) < 30 or d[gcol].nunique() < 10:
            print("Skipping: insufficient data")
            continue

        metrics = evaluate_grouped(d, setting, gcol, args.n_splits, args.test_size, args.seed)
        metrics.to_csv(out / "split_metrics.csv", index=False)

        rowwise = evaluate_rowwise(d, setting, args.n_splits, args.test_size, args.seed)
        rowwise.to_csv(out / "rowwise_split_metrics_sensitivity.csv", index=False)
        summary = summarize_split_metrics(metrics)
        summary.insert(0, "setting", setting)
        summary.insert(1, "group_mode", args.group_mode)
        summary.to_csv(out / "metric_summary.csv", index=False)
        row_summary = summarize_split_metrics(rowwise)
        row_summary.insert(0, "setting", setting)
        row_summary.insert(1, "group_mode", "rowwise_sensitivity")
        row_summary.to_csv(out / "rowwise_metric_summary_sensitivity.csv", index=False)
        all_summary.extend([summary, row_summary])

        coef = cluster_robust_full_fit(d, setting, gcol)
        coef.to_csv(out / "cluster_robust_coefficients.csv", index=False)

        pi = evaluate_conformal(d, setting, gcol, args.n_splits, args.test_size, args.seed)
        pi.to_csv(out / "conformal_interval_metrics.csv", index=False)
        if len(pi):
            pi_summary = summarize_split_metrics(
                pi, ["coverage", "q_log", "log_interval_width",
                     "multiplicative_half_width", "median_interval_width_kwh"]
            )
            pi_summary.to_csv(out / "conformal_interval_summary.csv", index=False)

        print(summary.to_string(index=False))
        if len(pi):
            print("Conformal coverage median:", pi["coverage"].median())

    if all_summary:
        pd.concat(all_summary, ignore_index=True).to_csv(root / "all_settings_summary.csv", index=False)


if __name__ == "__main__":
    main()
