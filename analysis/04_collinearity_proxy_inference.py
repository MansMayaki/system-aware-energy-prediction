#!/usr/bin/env python3
"""Collinearity diagnostics, cluster-aware inference, and memory-proxy robustness.

Reviewer coverage:
  - R1 comments 4, 7, 19, 20 and minor comment 18.
  - Reviewer 2 comments 1 and 3.

The proxy analysis does *not* claim to measure physical HBM traffic. It only
checks whether the predictive sign/fit is stable to several plausible workload
memory descriptors constructed from already available configuration variables.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from common import (
    add_group_ids, ensure_outdir, filter_hardware, full_formula, group_column,
    load_data, prediction_metrics, subset_setting, summarize_split_metrics,
    valid_group_splits,
)


def add_proxy_variants(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Recover the effective S used in the primary M_proxy whenever possible.
    required = ["M_proxy", "batch_size", "hidden_size", "num_layers"]
    if not all(c in d.columns for c in required):
        return d
    denom = d["batch_size"] * d["hidden_size"] * d["num_layers"]
    s_eff = d["M_proxy"] / denom.replace(0, np.nan)
    if "seq_len_batch" in d.columns:
        s_eff = s_eff.fillna(d["seq_len_batch"])
    if "max_length" in d.columns:
        s_eff = s_eff.fillna(d["max_length"])
    d["S_eff_proxy"] = s_eff

    B, S, width, L = d["batch_size"], d["S_eff_proxy"], d["hidden_size"], d["num_layers"]
    M1 = B * S * width * L
    # Attention-state pressure: add a sequence-quadratic term. This is still a
    # descriptor, not byte traffic.
    M2 = M1 + B * (S ** 2) * L
    if "params_calc" in d.columns:
        P = pd.to_numeric(d["params_calc"], errors="coerce")
        M3 = M1 + P
        M4 = M1 + P * (pd.to_numeric(d["gpus"], errors="coerce") > 1).astype(float)
    else:
        M3 = M1.copy(); M4 = M1.copy()

    for name, x in {
        "M1_activation": M1,
        "M2_activation_attention": M2,
        "M3_activation_parameter": M3,
        "M4_activation_distributed_state": M4,
    }.items():
        d[name] = x
        d["log_" + name] = np.where(x > 0, np.log(x), np.nan)
    return d


def vif_table(d: pd.DataFrame, cols):
    x = d[list(cols)].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(x) < 5:
        return pd.DataFrame()
    # Standardize before condition-number reporting; VIF itself is scale invariant.
    X = sm.add_constant(x)
    rows = []
    for i, c in enumerate(X.columns):
        if c == "const":
            continue
        rows.append({"variable": c, "VIF": variance_inflation_factor(X.values, i)})
    return pd.DataFrame(rows)


def cluster_vs_hc3(d: pd.DataFrame, setting: str, group_col: str):
    x = d.copy()
    for c in ["log_C", "log_M", "log_eta"]:
        x[c + "_c"] = x[c] - x[c].mean()
    formula = full_formula(setting)
    base = smf.ols(formula, data=x)
    hc3 = base.fit(cov_type="HC3")
    cl = base.fit(cov_type="cluster", cov_kwds={"groups": x[group_col]})
    terms = sorted(set(hc3.params.index) | set(cl.params.index))
    rows = []
    for t in terms:
        rows.append({
            "term": t,
            "coef": cl.params.get(t, np.nan),
            "HC3_SE": hc3.bse.get(t, np.nan),
            "cluster_SE": cl.bse.get(t, np.nan),
            "HC3_p": hc3.pvalues.get(t, np.nan),
            "cluster_p": cl.pvalues.get(t, np.nan),
            "n": len(x), "n_clusters": x[group_col].nunique(),
        })
    return pd.DataFrame(rows)


def proxy_grouped_validation(d: pd.DataFrame, setting: str, group_col: str,
                             n_splits: int, test_size: float, seed: int):
    proxy_cols = [
        ("primary_M_proxy", "log_M"),
        ("M1_activation", "log_M1_activation"),
        ("M2_activation_attention", "log_M2_activation_attention"),
        ("M3_activation_parameter", "log_M3_activation_parameter"),
        ("M4_activation_distributed_state", "log_M4_activation_distributed_state"),
    ]
    rows = []
    cats = ["strategy"] + (["arch"] if setting == "pooled" else [])
    split_defs = list(valid_group_splits(
        d, group_col, n_splits=n_splits, test_size=test_size,
        seed=seed, categorical_cols=cats,
    ))

    for proxy_name, pcol in proxy_cols:
        if pcol not in d.columns:
            continue
        for split_id, split_seed, tr_idx, te_idx in split_defs:
            train, test = d.iloc[tr_idx].copy(), d.iloc[te_idx].copy()
            needed = ["log_energy", "log_C", pcol, "log_eta", "strategy"]
            train = train.dropna(subset=needed); test = test.dropna(subset=needed)
            if len(train) < 20 or len(test) < 5:
                continue
            # Center using training statistics only.
            for c in ["log_C", pcol, "log_eta"]:
                mu = train[c].mean()
                train[c + "_c"] = train[c] - mu
                test[c + "_c"] = test[c] - mu
            terms = ["log_C_c", f"{pcol}_c", "log_eta_c",
                     'C(strategy, Treatment(reference="Baseline"))']
            if setting == "pooled":
                terms.append("C(arch)")
            formula = "log_energy ~ " + " + ".join(terms)
            m = smf.ols(formula, data=train).fit(cov_type="HC3")
            pred = m.predict(test)
            met = prediction_metrics(test["log_energy"], pred)
            rows.append({
                "setting": setting, "proxy": proxy_name, "split": split_id,
                **met,
                "alpha_C": m.params.get("log_C_c", np.nan),
                "alpha_M": m.params.get(f"{pcol}_c", np.nan),
                "alpha_eta": m.params.get("log_eta_c", np.nan),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="outputs/revision/diagnostics")
    ap.add_argument("--settings", nargs="+", default=["bert", "gpt", "pooled"])
    ap.add_argument("--group-mode", default="config_family",
                    choices=["config_family", "workload_family"])
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--hardware", nargs="*", default=None)
    args = ap.parse_args()

    root = ensure_outdir(args.outdir)
    data = add_proxy_variants(filter_hardware(add_group_ids(load_data(args.input)), args.hardware))
    gcol = group_column(args.group_mode)

    all_proxy = []
    for setting in args.settings:
        out = ensure_outdir(root / setting)
        d = subset_setting(data, setting).copy()
        d = d.dropna(subset=["log_energy", "log_C", "log_M", "log_eta", gcol])
        if len(d) < 30:
            continue

        # Correlation matrices among principal continuous regressors.
        cols = ["log_C", "log_M", "log_eta", "log2_gpus"]
        cols = [c for c in cols if c in d.columns]
        d[cols].corr(method="pearson").to_csv(out / "correlation_pearson.csv")
        d[cols].corr(method="spearman").to_csv(out / "correlation_spearman.csv")
        vif_table(d, ["log_C", "log_M", "log_eta"]).to_csv(out / "vif.csv", index=False)

        # Standardized condition number for the three core continuous predictors.
        x = d[["log_C", "log_M", "log_eta"]].dropna().copy()
        z = (x - x.mean()) / x.std(ddof=0).replace(0, 1)
        cond = np.linalg.cond(np.column_stack([np.ones(len(z)), z.to_numpy()]))
        pd.DataFrame([{"standardized_condition_number": cond, "n": len(z)}]).to_csv(
            out / "condition_number.csv", index=False
        )

        cluster_vs_hc3(d, setting, gcol).to_csv(out / "hc3_vs_cluster_se.csv", index=False)

        proxy = proxy_grouped_validation(
            d, setting, gcol, args.n_splits, args.test_size, args.seed
        )
        proxy.to_csv(out / "memory_proxy_split_results.csv", index=False)
        if len(proxy):
            all_proxy.append(proxy)
            compact = (
                proxy.groupby("proxy")
                .agg(
                    R2_median=("r2", "median"),
                    R2_q25=("r2", lambda s: s.quantile(.25)),
                    R2_q75=("r2", lambda s: s.quantile(.75)),
                    RMSE_median=("rmse", "median"),
                    alpha_C_median=("alpha_C", "median"),
                    alpha_M_median=("alpha_M", "median"),
                    alpha_eta_median=("alpha_eta", "median"),
                    n_splits=("split", "count"),
                ).reset_index()
            )
            compact.to_csv(out / "memory_proxy_summary.csv", index=False)
            print(f"\n[{setting}] memory-proxy robustness")
            print(compact.to_string(index=False))

    if all_proxy:
        pd.concat(all_proxy, ignore_index=True).to_csv(root / "all_proxy_results.csv", index=False)


if __name__ == "__main__":
    main()
