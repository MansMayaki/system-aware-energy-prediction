#!/usr/bin/env python3
"""Leakage-free pre-run efficiency and energy prediction.

Outer split: configuration-family disjoint train/test.
Inner split: configuration-family GroupKFold used only inside outer training to
create out-of-fold eta_hat values for the energy model.

The eta predictor uses only quantities known before the target execution:
  log_C, log_M, GPU count, and strategy (plus arch in the pooled case).
It never uses target runtime, energy, throughput, utilization, temperature, or
other post-run telemetry.

Outputs:
  point_metrics.csv / point_summary.csv
  eta_feature_importance.csv
  conformal_metrics.csv / conformal_summary.csv
  optional scale_shift_metrics.csv

Reviewer coverage: R1 comments 5, 6, 15; questions 6, 7, 11.
Reviewer 2 comment 4: indirect-leakage concern, feature importance, strict
architecture/grouped validation, and scale-shift stress tests.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    add_group_ids, conformal_q, ensure_outdir, filter_hardware, group_column,
    load_data, onehot_encoder_dense, prediction_metrics, subset_setting,
    summarize_split_metrics, valid_group_splits,
)


def eta_feature_sets(setting: str):
    # Principal model intentionally stays small and transparent, matching the
    # original manuscript contribution. All variables are pre-run.
    numeric = ["log_C", "log_M", "gpus"]
    categorical = ["strategy"] + (["arch"] if setting == "pooled" else [])
    return numeric, categorical


def make_eta_model(numeric: List[str], categorical: List[str]):
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", onehot_encoder_dense()),
    ])
    pre = ColumnTransformer([
        ("num", num, numeric),
        ("cat", cat, categorical),
    ], remainder="drop")
    return Pipeline([("pre", pre), ("reg", LinearRegression())])


def safe_group_kfold_indices(df: pd.DataFrame, group_col: str, n_splits: int = 5):
    ng = df[group_col].nunique()
    k = min(n_splits, ng)
    if k < 2:
        raise ValueError("Need at least two groups for inner cross-fitting.")
    gkf = GroupKFold(n_splits=k)
    return list(gkf.split(df, groups=df[group_col]))


def crossfit_eta(train: pd.DataFrame, group_col: str, numeric: List[str], categorical: List[str]):
    """OOF log-eta predictions for outer-training rows + final eta model."""
    features = numeric + categorical
    oof = pd.Series(np.nan, index=train.index, dtype=float)

    for inner_tr, inner_va in safe_group_kfold_indices(train, group_col, n_splits=5):
        tr = train.iloc[inner_tr]
        va = train.iloc[inner_va]
        model = make_eta_model(numeric, categorical)
        model.fit(tr[features], tr["log_eta"])
        oof.loc[va.index] = model.predict(va[features])

    if oof.isna().any():
        raise RuntimeError("Cross-fitted eta predictions contain missing values.")

    final_model = make_eta_model(numeric, categorical)
    final_model.fit(train[features], train["log_eta"])
    return oof, final_model


def energy_formula(setting: str, eta_name: str):
    terms = ["log_C_c", "log_M_c", f"{eta_name}_c", 'C(strategy, Treatment(reference="Baseline"))']
    if setting == "pooled":
        terms.append("C(arch)")
    return "log_energy ~ " + " + ".join(terms)


def config_only_formula(setting: str, include_strategy: bool = True):
    terms = ["log_C_c", "log_M_c", "gpus_c"]
    if include_strategy:
        terms.append('C(strategy, Treatment(reference="Baseline"))')
    if setting == "pooled":
        terms.append("C(arch)")
    return "log_energy ~ " + " + ".join(terms)


def measured_eta_formula(setting: str):
    return energy_formula(setting, "log_eta")


def center_pair(train: pd.DataFrame, test: pd.DataFrame, cols):
    train = train.copy(); test = test.copy()
    for c in cols:
        mu = float(train[c].mean())
        train[c + "_c"] = train[c] - mu
        test[c + "_c"] = test[c] - mu
    return train, test


def fit_energy_predicted_eta(setting: str, train: pd.DataFrame, test: pd.DataFrame,
                             eta_oof: pd.Series, eta_test: np.ndarray):
    tr = train.copy(); te = test.copy()
    tr["log_eta_hat"] = eta_oof.loc[tr.index].to_numpy()
    te["log_eta_hat"] = eta_test
    tr, te = center_pair(tr, te, ["log_C", "log_M", "log_eta_hat"])
    m = smf.ols(energy_formula(setting, "log_eta_hat"), data=tr).fit(cov_type="HC3")
    return m, np.asarray(m.predict(te), dtype=float)


def fit_energy_config_only(setting: str, train: pd.DataFrame, test: pd.DataFrame, include_strategy: bool = True):
    tr, te = center_pair(train, test, ["log_C", "log_M", "gpus"])
    m = smf.ols(config_only_formula(setting, include_strategy=include_strategy), data=tr).fit(cov_type="HC3")
    return m, np.asarray(m.predict(te), dtype=float)


def fit_energy_measured_eta(setting: str, train: pd.DataFrame, test: pd.DataFrame):
    tr, te = center_pair(train, test, ["log_C", "log_M", "log_eta"])
    m = smf.ols(measured_eta_formula(setting), data=tr).fit(cov_type="HC3")
    return m, np.asarray(m.predict(te), dtype=float)


def evaluate_point_predictions(df: pd.DataFrame, setting: str, group_col: str,
                               n_splits: int, test_size: float, seed: int):
    numeric, categorical = eta_feature_sets(setting)
    features = numeric + categorical
    rows, imp_rows = [], []
    cat_split = ["strategy"] + (["arch"] if setting == "pooled" else [])

    for split_id, split_seed, tr_idx, te_idx in valid_group_splits(
        df, group_col, n_splits=n_splits, test_size=test_size,
        seed=seed, categorical_cols=cat_split,
    ):
        train = df.iloc[tr_idx].copy()
        test = df.iloc[te_idx].copy()

        eta_oof, eta_model = crossfit_eta(train, group_col, numeric, categorical)
        eta_test = eta_model.predict(test[features])

        # 1) eta prediction itself
        eta_met = prediction_metrics(test["log_eta"], eta_test)
        rows.append({"setting": setting, "split": split_id, "model": "eta_predictor",
                     **eta_met})

        # 2a) Legacy configuration-only baseline from the submitted manuscript:
        #     logC + logM + N, intentionally omitting strategy. Kept only to
        #     reproduce the historical comparison.
        _, pred_cfg_legacy = fit_energy_config_only(setting, train, test, include_strategy=False)
        cfg_legacy_met = prediction_metrics(test["log_energy"], pred_cfg_legacy)
        rows.append({"setting": setting, "split": split_id, "model": "configuration_only_legacy_no_strategy",
                     **cfg_legacy_met})

        # 2b) Fair matched-feature pre-run baseline. This includes strategy,
        #     because strategy is itself known before execution and is also used
        #     by the eta predictor. This is the comparison that should be used in
        #     the revision when discussing incremental predictive accuracy.
        _, pred_cfg = fit_energy_config_only(setting, train, test, include_strategy=True)
        cfg_met = prediction_metrics(test["log_energy"], pred_cfg)
        rows.append({"setting": setting, "split": split_id, "model": "configuration_only_matched_features",
                     **cfg_met})

        # 3) proposed leakage-free predicted-eta pre-run energy model
        _, pred_pre = fit_energy_predicted_eta(setting, train, test, eta_oof, eta_test)
        pre_met = prediction_metrics(test["log_energy"], pred_pre)
        rows.append({"setting": setting, "split": split_id, "model": "predicted_eta_prerun",
                     **pre_met})

        # 4) post-run measured-eta reference/upper bound
        _, pred_meas = fit_energy_measured_eta(setting, train, test)
        meas_met = prediction_metrics(test["log_energy"], pred_meas)
        rows.append({"setting": setting, "split": split_id, "model": "measured_eta_postrun",
                     **meas_met})

        # Held-out permutation importance for eta predictor. Since permutation is
        # applied to the raw columns before the pipeline, values correspond to the
        # original pre-run variables rather than one-hot-expanded columns.
        try:
            pi = permutation_importance(
                eta_model, test[features], test["log_eta"], scoring="r2",
                n_repeats=30, random_state=seed + split_id, n_jobs=-1,
            )
            for feat, mean_imp, std_imp in zip(features, pi.importances_mean, pi.importances_std):
                imp_rows.append({
                    "setting": setting, "split": split_id, "feature": feat,
                    "importance_r2_drop": float(mean_imp),
                    "importance_std": float(std_imp),
                })
        except Exception as e:
            print(f"Permutation importance failed on {setting} split {split_id}: {e}")

    return pd.DataFrame(rows), pd.DataFrame(imp_rows)


def conformal_one_split(df: pd.DataFrame, setting: str, group_col: str,
                        outer_train_idx, test_idx, seed: int, alpha=0.05,
                        calib_frac=0.20):
    numeric, categorical = eta_feature_sets(setting)
    features = numeric + categorical
    outer_train = df.iloc[outer_train_idx].copy()
    test = df.iloc[test_idx].copy()

    gss = GroupShuffleSplit(n_splits=1, test_size=calib_frac, random_state=seed)
    fit_rel, cal_rel = next(gss.split(outer_train, groups=outer_train[group_col]))
    fit = outer_train.iloc[fit_rel].copy()
    cal = outer_train.iloc[cal_rel].copy()

    # Ensure factor levels needed for statsmodels are represented in fit.
    for c in ["strategy"] + (["arch"] if setting == "pooled" else []):
        if not set(cal[c]).issubset(set(fit[c])) or not set(test[c]).issubset(set(fit[c])):
            return None

    eta_oof_fit, eta_model = crossfit_eta(fit, group_col, numeric, categorical)
    eta_cal = eta_model.predict(cal[features])
    eta_test = eta_model.predict(test[features])

    energy_model, pred_cal = fit_energy_predicted_eta(setting, fit, cal, eta_oof_fit, eta_cal)
    # Reconstruct test centering from fit by fitting through the same helper.
    _, pred_test = fit_energy_predicted_eta(setting, fit, test, eta_oof_fit, eta_test)

    residuals = np.abs(cal["log_energy"].to_numpy() - pred_cal)
    q = conformal_q(residuals, alpha=alpha)
    y = test["log_energy"].to_numpy()
    lo, hi = pred_test - q, pred_test + q
    cov = np.mean((y >= lo) & (y <= hi))

    met = prediction_metrics(y, pred_test)
    met.update({
        "coverage": float(cov),
        "q_log": float(q),
        "multiplicative_half_width": float(np.exp(q)),
        "median_interval_width_kwh": float(np.median(np.exp(hi) - np.exp(lo))),
        "n_fit": len(fit), "n_cal": len(cal), "n_test": len(test),
        "n_fit_groups": fit[group_col].nunique(),
        "n_cal_groups": cal[group_col].nunique(),
        "n_test_groups": test[group_col].nunique(),
    })
    return met


def evaluate_conformal(df, setting, group_col, n_splits, test_size, seed):
    rows = []
    cats = ["strategy"] + (["arch"] if setting == "pooled" else [])
    for split_id, split_seed, tr_idx, te_idx in valid_group_splits(
        df, group_col, n_splits=n_splits, test_size=test_size,
        seed=seed + 5000, categorical_cols=cats,
    ):
        result = None
        for j in range(50):
            result = conformal_one_split(
                df, setting, group_col, tr_idx, te_idx,
                seed=split_seed + j, alpha=0.05, calib_frac=0.20,
            )
            if result is not None:
                break
        if result is not None:
            result.update({"setting": setting, "split": split_id})
            rows.append(result)
    return pd.DataFrame(rows)


def scale_shift_stress_test(df: pd.DataFrame, setting: str, group_col: str):
    """Simple distribution-shift stress test on model/workload scale.

    Families are ranked by median log_C. Two extrapolation directions are tested:
      small -> large: train on lower 75%, test highest 25%
      large -> small: train on upper 75%, test lowest 25%
    The split is family-disjoint by construction.
    """
    numeric, categorical = eta_feature_sets(setting)
    features = numeric + categorical
    family_scale = df.groupby(group_col)["log_C"].median().sort_values()
    q25, q75 = family_scale.quantile([0.25, 0.75])
    scenarios = {
        "small_to_large": (family_scale[family_scale <= q75].index,
                           family_scale[family_scale > q75].index),
        "large_to_small": (family_scale[family_scale >= q25].index,
                           family_scale[family_scale < q25].index),
    }
    rows = []
    for name, (tr_groups, te_groups) in scenarios.items():
        train = df[df[group_col].isin(tr_groups)].copy()
        test = df[df[group_col].isin(te_groups)].copy()
        if len(train) < 20 or len(test) < 5 or train[group_col].nunique() < 5:
            continue
        # Require categories in test to exist in train for statsmodels energy fit.
        valid = True
        for c in ["strategy"] + (["arch"] if setting == "pooled" else []):
            if not set(test[c]).issubset(set(train[c])):
                valid = False
        if not valid:
            continue
        eta_oof, eta_model = crossfit_eta(train, group_col, numeric, categorical)
        eta_test = eta_model.predict(test[features])
        _, pred = fit_energy_predicted_eta(setting, train, test, eta_oof, eta_test)
        met = prediction_metrics(test["log_energy"], pred)
        eta_met = prediction_metrics(test["log_eta"], eta_test)
        rows.append({
            "setting": setting, "scenario": name,
            "n_train": len(train), "n_test": len(test),
            "n_train_groups": train[group_col].nunique(),
            "n_test_groups": test[group_col].nunique(),
            "energy_r2": met["r2"], "energy_rmse": met["rmse"],
            "energy_calib_slope": met["calib_slope"],
            "eta_r2": eta_met["r2"], "eta_rmse": eta_met["rmse"],
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="outputs/revision/prerun")
    ap.add_argument("--settings", nargs="+", default=["bert", "gpt", "pooled"])
    ap.add_argument("--group-mode", default="config_family",
                    choices=["config_family", "workload_family"])
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--hardware", nargs="*", default=None)
    ap.add_argument("--skip-scale-shift", action="store_true")
    args = ap.parse_args()

    root = ensure_outdir(args.outdir)
    data = filter_hardware(add_group_ids(load_data(args.input)), args.hardware)
    gcol = group_column(args.group_mode)

    all_point, all_imp, all_conf, all_shift = [], [], [], []
    for setting in args.settings:
        d = subset_setting(data, setting).copy()
        numeric, categorical = eta_feature_sets(setting)
        needed = ["log_energy", "log_eta", gcol] + numeric + categorical
        d = d.dropna(subset=[c for c in needed if c in d.columns]).copy()
        print(f"\n[{setting}] rows={len(d)}, groups={d[gcol].nunique()}")
        if len(d) < 30 or d[gcol].nunique() < 10:
            print("Skipping: insufficient data")
            continue

        point, imp = evaluate_point_predictions(
            d, setting, gcol, args.n_splits, args.test_size, args.seed
        )
        all_point.append(point)
        if len(imp): all_imp.append(imp)

        conf = evaluate_conformal(d, setting, gcol, args.n_splits, args.test_size, args.seed)
        if len(conf): all_conf.append(conf)

        if not args.skip_scale_shift:
            shift = scale_shift_stress_test(d, setting, gcol)
            if len(shift): all_shift.append(shift)

    point = pd.concat(all_point, ignore_index=True)
    point.to_csv(root / "point_metrics.csv", index=False)

    summaries = []
    for (setting, model), sub in point.groupby(["setting", "model"]):
        s = summarize_split_metrics(sub)
        s.insert(0, "model", model); s.insert(0, "setting", setting)
        summaries.append(s)
    pd.concat(summaries, ignore_index=True).to_csv(root / "point_summary.csv", index=False)

    if all_imp:
        imp = pd.concat(all_imp, ignore_index=True)
        imp.to_csv(root / "eta_feature_importance_split.csv", index=False)
        imp_summary = (
            imp.groupby(["setting", "feature"])["importance_r2_drop"]
            .agg(["median", "mean", "std", "count"]).reset_index()
            .sort_values(["setting", "median"], ascending=[True, False])
        )
        imp_summary.to_csv(root / "eta_feature_importance.csv", index=False)

    if all_conf:
        conf = pd.concat(all_conf, ignore_index=True)
        conf.to_csv(root / "conformal_metrics.csv", index=False)
        conf_summaries = []
        for setting, sub in conf.groupby("setting"):
            s = summarize_split_metrics(
                sub, ["r2", "rmse", "calib_slope", "calib_intercept",
                      "coverage", "q_log", "multiplicative_half_width",
                      "median_interval_width_kwh"]
            )
            s.insert(0, "setting", setting)
            conf_summaries.append(s)
        pd.concat(conf_summaries, ignore_index=True).to_csv(root / "conformal_summary.csv", index=False)

    if all_shift:
        pd.concat(all_shift, ignore_index=True).to_csv(root / "scale_shift_metrics.csv", index=False)

    print("\nSaved results to", root)


if __name__ == "__main__":
    main()
