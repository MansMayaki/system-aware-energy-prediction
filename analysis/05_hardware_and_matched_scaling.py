#!/usr/bin/env python3
"""Hardware sensitivity, strategy/GPU-count interactions, and matched scaling.

Reviewer coverage:
  R1 comments 8-10 and question 10.
  Reviewer 2 comments 2 and 6 (empirical part only; implementation metadata such
  as TP parallel dimension/kernel fusion must still be documented from code/logs).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import (
    CONFIG_FAMILY_COLUMNS, add_group_ids, ensure_outdir, filter_hardware,
    load_data, prediction_metrics, subset_setting, valid_group_splits,
)


def bootstrap_median_ci(x, n_boot=4000, seed=2026, alpha=0.05):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    meds = np.empty(n_boot)
    for b in range(n_boot):
        meds[b] = np.median(rng.choice(x, size=len(x), replace=True))
    return float(np.median(x)), float(np.quantile(meds, alpha/2)), float(np.quantile(meds, 1-alpha/2))


def matched_scaling(df: pd.DataFrame, n_boot=4000, seed=2026):
    d = df.copy()
    needed = ["gpus", "strategy", "energy_consumed_kWh", "duration_s", "tokens_seen"]
    d = d.dropna(subset=needed)
    d = d[(d["duration_s"] > 0) & (d["tokens_seen"] > 0) & (d["energy_consumed_kWh"] > 0)]
    d["throughput"] = d["tokens_seen"] / d["duration_s"]
    d["energy_per_token"] = d["energy_consumed_kWh"] / d["tokens_seen"]

    keys = [c for c in CONFIG_FAMILY_COLUMNS if c in d.columns]
    base = d[(d["gpus"] == 1) & (d["strategy"] == "Baseline")].copy()
    if len(base) == 0:
        base = d[d["gpus"] == 1].copy()

    # If multiple baseline rows remain for the same matching key, aggregate them
    # before merging. This avoids arbitrary matching.
    base = (
        base.groupby(keys, dropna=False)
        .agg(
            baseline_throughput=("throughput", "median"),
            baseline_energy_per_token=("energy_per_token", "median"),
            baseline_energy=("energy_consumed_kWh", "median"),
            baseline_duration=("duration_s", "median"),
            baseline_tokens=("tokens_seen", "median"),
        ).reset_index()
    )

    multi = d[d["gpus"] > 1].copy()
    pairs = multi.merge(base, on=keys, how="inner", validate="many_to_one")
    pairs["throughput_gain"] = pairs["throughput"] / pairs["baseline_throughput"]
    pairs["matched_parallel_efficiency"] = pairs["throughput_gain"] / pairs["gpus"]
    pairs["energy_per_token_ratio"] = pairs["energy_per_token"] / pairs["baseline_energy_per_token"]
    pairs["energy_per_token_change_pct"] = 100 * (pairs["energy_per_token_ratio"] - 1)

    groups = [c for c in ["gpu_model", "arch", "strategy", "gpus"] if c in pairs.columns]
    rows = []
    for gi, (key, sub) in enumerate(pairs.groupby(groups, dropna=False)):
        if not isinstance(key, tuple): key = (key,)
        row = dict(zip(groups, key))
        row["n_pairs"] = len(sub)
        for col, prefix in [
            ("throughput_gain", "throughput_gain"),
            ("matched_parallel_efficiency", "parallel_efficiency"),
            ("energy_per_token_change_pct", "energy_per_token_change_pct"),
        ]:
            med, lo, hi = bootstrap_median_ci(sub[col], n_boot=n_boot, seed=seed + gi)
            row[prefix + "_median"] = med
            row[prefix + "_ci_low"] = lo
            row[prefix + "_ci_high"] = hi
        rows.append(row)
    return pairs, pd.DataFrame(rows)


def hardware_specific_grouped(df: pd.DataFrame, setting: str, n_splits: int, test_size: float, seed: int):
    rows = []
    coef_rows = []
    for gpu, sub in df.groupby("gpu_model"):
        sub = sub.copy()
        if len(sub) < 30 or sub["config_family_id"].nunique() < 10 or sub["strategy"].nunique() < 2:
            continue
        needed = ["log_energy", "log_C", "log_M", "log_eta", "strategy"]
        sub = sub.dropna(subset=needed)
        split_defs = list(valid_group_splits(
            sub, "config_family_id", n_splits=n_splits, test_size=test_size,
            seed=seed, categorical_cols=["strategy"],
        ))
        for split_id, split_seed, tr_idx, te_idx in split_defs:
            tr, te = sub.iloc[tr_idx].copy(), sub.iloc[te_idx].copy()
            for c in ["log_C", "log_M", "log_eta"]:
                mu = tr[c].mean(); tr[c + "_c"] = tr[c] - mu; te[c + "_c"] = te[c] - mu
            f = 'log_energy ~ log_C_c + log_M_c + log_eta_c + C(strategy, Treatment(reference="Baseline"))'
            m = smf.ols(f, data=tr).fit(cov_type="HC3")
            pred = m.predict(te)
            met = prediction_metrics(te["log_energy"], pred)
            rows.append({"setting": setting, "gpu_model": gpu, "split": split_id, **met})

        # Full-data coefficient calibration within hardware, cluster-aware.
        x = sub.copy()
        for c in ["log_C", "log_M", "log_eta"]:
            x[c + "_c"] = x[c] - x[c].mean()
        f = 'log_energy ~ log_C_c + log_M_c + log_eta_c + C(strategy, Treatment(reference="Baseline"))'
        m = smf.ols(f, data=x).fit(cov_type="cluster", cov_kwds={"groups": x["config_family_id"]})
        for term in ["log_C_c", "log_M_c", "log_eta_c", 'C(strategy, Treatment(reference="Baseline"))[T.FSDP]', 'C(strategy, Treatment(reference="Baseline"))[T.TP]']:
            if term in m.params:
                coef_rows.append({
                    "setting": setting, "gpu_model": gpu, "term": term,
                    "coef": m.params[term], "cluster_se": m.bse[term],
                    "pvalue": m.pvalues[term], "n": len(x),
                    "n_groups": x["config_family_id"].nunique(),
                })
    return pd.DataFrame(rows), pd.DataFrame(coef_rows)


def leave_one_hardware_out(df: pd.DataFrame, setting: str):
    """Stress test only; coefficients are expected to require recalibration."""
    rows = []
    if df["gpu_model"].nunique() < 2:
        return pd.DataFrame()
    for gpu in sorted(df["gpu_model"].dropna().unique()):
        tr = df[df["gpu_model"] != gpu].copy()
        te = df[df["gpu_model"] == gpu].copy()
        needed = ["log_energy", "log_C", "log_M", "log_eta", "strategy"]
        tr = tr.dropna(subset=needed); te = te.dropna(subset=needed)
        if len(tr) < 30 or len(te) < 10:
            continue
        if not set(te["strategy"]).issubset(set(tr["strategy"])):
            continue
        if setting == "pooled" and not set(te["arch"]).issubset(set(tr["arch"])):
            continue
        for c in ["log_C", "log_M", "log_eta"]:
            mu = tr[c].mean(); tr[c + "_c"] = tr[c] - mu; te[c + "_c"] = te[c] - mu
        terms = ["log_C_c", "log_M_c", "log_eta_c", 'C(strategy, Treatment(reference="Baseline"))']
        if setting == "pooled": terms.append("C(arch)")
        m = smf.ols("log_energy ~ " + " + ".join(terms), data=tr).fit()
        pred = m.predict(te)
        met = prediction_metrics(te["log_energy"], pred)
        rows.append({
            "setting": setting, "held_out_gpu": gpu,
            "n_train": len(tr), "n_test": len(te), **met
        })
    return pd.DataFrame(rows)


def interaction_fit(df: pd.DataFrame, setting: str):
    d = df.dropna(subset=["log_energy", "log_C", "log_M", "log_eta", "log2_gpus", "strategy"]).copy()
    for c in ["log_C", "log_M", "log_eta"]:
        d[c + "_c"] = d[c] - d[c].mean()
    terms = [
        "log_C_c", "log_M_c", "log_eta_c", "log2_gpus",
        'C(strategy, Treatment(reference="Baseline"))',
        'C(strategy, Treatment(reference="Baseline")):log2_gpus',
    ]
    if setting == "pooled": terms.append("C(arch)")
    if "gpu_model" in d.columns: terms.append("C(gpu_model)")
    m = smf.ols("log_energy ~ " + " + ".join(terms), data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["config_family_id"]}
    )
    ci = m.conf_int()
    return pd.DataFrame({
        "term": m.params.index, "coef": m.params.values, "cluster_se": m.bse.values,
        "pvalue": m.pvalues.values, "ci_low": ci.iloc[:,0].values,
        "ci_high": ci.iloc[:,1].values,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="outputs/revision/hardware_scaling")
    ap.add_argument("--settings", nargs="+", default=["bert", "gpt", "pooled"])
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-bootstrap", type=int, default=4000)
    ap.add_argument("--hardware", nargs="*", default=None)
    args = ap.parse_args()

    out = ensure_outdir(args.outdir)
    data = filter_hardware(add_group_ids(load_data(args.input)), args.hardware)

    pairs, matched = matched_scaling(data, n_boot=args.n_bootstrap, seed=args.seed)
    pairs.to_csv(out / "matched_pairs.csv", index=False)
    matched.to_csv(out / "matched_scaling_summary.csv", index=False)

    hw_metrics, hw_coefs, loho, interactions = [], [], [], []
    for setting in args.settings:
        d = subset_setting(data, setting).copy()
        if len(d) < 30:
            continue
        m, c = hardware_specific_grouped(d, setting, args.n_splits, args.test_size, args.seed)
        if len(m): hw_metrics.append(m)
        if len(c): hw_coefs.append(c)
        l = leave_one_hardware_out(d, setting)
        if len(l): loho.append(l)
        inter = interaction_fit(d, setting)
        inter.insert(0, "setting", setting)
        interactions.append(inter)

    if hw_metrics: pd.concat(hw_metrics, ignore_index=True).to_csv(out / "hardware_specific_split_metrics.csv", index=False)
    if hw_coefs: pd.concat(hw_coefs, ignore_index=True).to_csv(out / "hardware_specific_coefficients.csv", index=False)
    if loho: pd.concat(loho, ignore_index=True).to_csv(out / "leave_one_hardware_out.csv", index=False)
    if interactions: pd.concat(interactions, ignore_index=True).to_csv(out / "strategy_gpu_count_interactions.csv", index=False)

    # TP-only empirical summary for Reviewer 2. The missing implementation-level
    # metadata must be recovered from the training code/logs, not inferred here.
    if len(pairs):
        tp = pairs[pairs["strategy"].eq("TP")].copy()
        if len(tp):
            cols = [c for c in ["gpu_model", "arch", "gpus", "throughput_gain",
                                "matched_parallel_efficiency", "energy_per_token_change_pct"] if c in tp.columns]
            tp[cols].to_csv(out / "tp_matched_pairs.csv", index=False)

    print("\nMatched scaling summary:")
    print(matched.to_string(index=False))


if __name__ == "__main__":
    main()
