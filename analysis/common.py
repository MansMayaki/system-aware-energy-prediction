#!/usr/bin/env python3
"""Shared utilities for the ARRAY major-revision analyses.

The canonical input is a *configuration-level* CSV: repeated executions of an
identical configuration have already been aggregated, and `n_runs` records the
number of raw repetitions represented by each row.

Nothing in this module uses target-run energy/runtime to construct pre-run
features. Runtime-derived quantities are kept separate and are used only in
post-run analyses.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder

EPS = 1e-12

# Same workload/hardware configuration, but deliberately excludes strategy and
# GPU count. Hence Baseline/FSDP/TP and N=1/2/4/... variants of the same workload
# can never be split between train and test.
CONFIG_FAMILY_COLUMNS = [
    "gpu_model", "arch", "model", "max_length", "batch_size", "seq_len_batch",
    "grad_accum", "num_layers", "hidden_size", "heads", "ff", "task",
    "fp16", "bf16", "power_correction",
]

# Stricter sensitivity grouping: excludes hardware too, so an architectural /
# workload configuration observed on one GPU family cannot appear in the other
# partition on another GPU family.
WORKLOAD_FAMILY_COLUMNS = [
    "arch", "model", "max_length", "batch_size", "seq_len_batch", "grad_accum",
    "num_layers", "hidden_size", "heads", "ff", "task", "fp16", "bf16",
    "power_correction",
]

# Exact row-level configuration key. This includes the execution choices that
# CONFIG_FAMILY_COLUMNS intentionally omit.
EXACT_CONFIG_COLUMNS = CONFIG_FAMILY_COLUMNS + ["strategy", "gpus"]


def _safe_log(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    m = np.isfinite(x) & (x > 0)
    out.loc[m] = np.log(x.loc[m])
    return out


def load_data(path: str | Path) -> pd.DataFrame:
    """Load and normalize the final configuration-level analysis table."""
    df = pd.read_csv(path)
    unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    if "GPU model" in df.columns and "gpu_model" not in df.columns:
        df = df.rename(columns={"GPU model": "gpu_model"})

    required = ["arch", "strategy", "gpus", "energy_consumed_kWh"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["arch"] = df["arch"].astype(str).str.strip().str.lower()
    df["strategy"] = (
        df["strategy"].astype(str).str.strip()
        .replace({"FSDP_2d": "FSDP", "TP_2d": "TP"})
    )
    if "gpu_model" in df.columns:
        df["gpu_model"] = df["gpu_model"].astype(str).str.strip()

    numeric_candidates = [
        "gpus", "energy_consumed_kWh", "duration_s", "tokens_seen", "FLOPs",
        "params_calc", "M_proxy", "compute", "memory", "parallel_efficiency",
        "speed_up", "batch_size", "max_length", "seq_len_batch", "grad_accum",
        "num_layers", "hidden_size", "heads", "ff", "n_runs",
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Canonical logs used throughout the revision.
    log_map = {
        "energy_consumed_kWh": "log_energy",
        "compute": "log_C",
        "M_proxy": "log_M",
        "parallel_efficiency": "log_eta",
        "duration_s": "log_duration",
        "tokens_seen": "log_tokens",
        "FLOPs": "log_total_flops",
    }
    for raw, new in log_map.items():
        if new not in df.columns and raw in df.columns:
            df[new] = _safe_log(df[raw])

    # Fall back to notebook column names if present.
    fallback = {
        "log_energy_consumed_kWh": "log_energy",
        "log_compute": "log_C",
        "log_M_proxy": "log_M",
        "log_parallel_efficiency": "log_eta",
        "log_tokens_seen": "log_tokens",
        "log_FLOPs": "log_total_flops",
    }
    for old, new in fallback.items():
        if new not in df.columns and old in df.columns:
            df[new] = pd.to_numeric(df[old], errors="coerce")

    if "duration_s" in df.columns:
        df["device_seconds"] = pd.to_numeric(df["duration_s"], errors="coerce") * pd.to_numeric(df["gpus"], errors="coerce")
        df["log_device_seconds"] = _safe_log(df["device_seconds"])
    df["log_gpus"] = _safe_log(df["gpus"])
    df["log2_gpus"] = np.log2(pd.to_numeric(df["gpus"], errors="coerce").clip(lower=EPS))

    return df


def _string_key(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        raise ValueError("No grouping columns are available in the input table.")
    z = df[cols].copy()
    for c in cols:
        z[c] = z[c].where(z[c].notna(), "<NA>").astype(str)
    return z.agg("|".join, axis=1)


def add_group_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["config_family_id"] = _string_key(df, CONFIG_FAMILY_COLUMNS)
    df["workload_family_id"] = _string_key(df, WORKLOAD_FAMILY_COLUMNS)
    df["exact_config_id"] = _string_key(df, EXACT_CONFIG_COLUMNS)
    return df


def group_column(mode: str) -> str:
    mode = mode.lower().strip()
    if mode in {"config", "config_family", "hardware_config"}:
        return "config_family_id"
    if mode in {"workload", "workload_family", "strict"}:
        return "workload_family_id"
    if mode in {"exact", "exact_config"}:
        return "exact_config_id"
    raise ValueError(f"Unknown group mode: {mode}")


def subset_setting(df: pd.DataFrame, setting: str) -> pd.DataFrame:
    setting = setting.lower()
    d = df.copy()
    if setting == "bert":
        d = d[d["arch"].eq("bert")]
        d = d[d["strategy"].isin(["Baseline", "FSDP", "TP"])]
    elif setting == "gpt":
        d = d[d["arch"].eq("gpt")]
        d = d[d["strategy"].isin(["Baseline", "FSDP"])]
    elif setting == "pooled":
        d = d[d["arch"].isin(["bert", "gpt"])]
        d = d[d["strategy"].isin(["Baseline", "FSDP"])]
    else:
        raise ValueError("setting must be one of: bert, gpt, pooled")
    return d.copy()


def filter_hardware(df: pd.DataFrame, hardware: Optional[Sequence[str]]) -> pd.DataFrame:
    if not hardware:
        return df
    if "gpu_model" not in df.columns:
        raise ValueError("Hardware filter requested but gpu_model is absent.")
    return df[df["gpu_model"].isin(list(hardware))].copy()


def prediction_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    if len(y_true) < 3:
        return {"n_test": len(y_true), "r2": np.nan, "rmse": np.nan,
                "calib_slope": np.nan, "calib_intercept": np.nan}
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    X = np.column_stack([np.ones(len(y_pred)), y_pred])
    beta, *_ = np.linalg.lstsq(X, y_true, rcond=None)
    return {
        "n_test": int(len(y_true)),
        "r2": float(r2),
        "rmse": float(rmse),
        "calib_intercept": float(beta[0]),
        "calib_slope": float(beta[1]),
    }


def summarize_split_metrics(df: pd.DataFrame, metric_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if metric_cols is None:
        metric_cols = ["r2", "rmse", "calib_slope", "calib_intercept"]
    rows = []
    for c in metric_cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) == 0:
            continue
        rows.append({
            "metric": c,
            "n_splits": int(len(s)),
            "median": float(s.median()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "p025": float(s.quantile(0.025)),
            "p975": float(s.quantile(0.975)),
        })
    return pd.DataFrame(rows)


def valid_group_splits(
    df: pd.DataFrame,
    group_col: str,
    n_splits: int = 10,
    test_size: float = 0.30,
    seed: int = 42,
    categorical_cols: Sequence[str] = ("strategy",),
    max_attempts: int = 1000,
):
    """Yield deterministic group-disjoint splits with categorical support in train.

    For every split, every categorical level appearing in test must also appear in
    train. This prevents formula/pipeline failures without leaking groups.
    """
    groups = df[group_col].astype(str).to_numpy()
    yielded = 0
    attempts = 0
    rng = np.random.default_rng(seed)
    while yielded < n_splits and attempts < max_attempts:
        rs = int(rng.integers(0, 2**31 - 1))
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        tr, te = next(gss.split(df, groups=groups))
        train, test = df.iloc[tr], df.iloc[te]
        okay = True
        for c in categorical_cols:
            if c not in df.columns:
                continue
            train_levels = set(train[c].dropna().astype(str))
            test_levels = set(test[c].dropna().astype(str))
            if not test_levels.issubset(train_levels):
                okay = False
                break
        if okay:
            yield yielded, rs, tr, te
            yielded += 1
        attempts += 1
    if yielded < n_splits:
        raise RuntimeError(f"Could only construct {yielded}/{n_splits} valid group splits.")


def add_centered(train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str]):
    train = train.copy()
    test = test.copy()
    means = {}
    for c in cols:
        mu = float(pd.to_numeric(train[c], errors="coerce").mean())
        means[c] = mu
        train[c + "_c"] = train[c] - mu
        test[c + "_c"] = test[c] - mu
    return train, test, means


def full_formula(setting: str, eta_col: str = "log_eta", include_hardware: bool = False,
                 interactions: bool = False) -> str:
    terms = ["log_C_c", "log_M_c", f"{eta_col}_c", 'C(strategy, Treatment(reference="Baseline"))']
    if setting == "pooled":
        terms.append("C(arch)")
    if include_hardware:
        terms.append("C(gpu_model)")
    if interactions:
        terms.extend(["log2_gpus", 'C(strategy, Treatment(reference="Baseline")):log2_gpus'])
    return "log_energy ~ " + " + ".join(terms)


def fit_statsmodels(train: pd.DataFrame, test: pd.DataFrame, formula: str,
                    center_cols: Sequence[str], cov_type: str = "HC3",
                    cluster_groups: Optional[pd.Series] = None):
    train_c, test_c, means = add_centered(train, test, center_cols)
    if cov_type == "cluster":
        if cluster_groups is None:
            raise ValueError("cluster_groups is required for cluster covariance")
        # align cluster labels to centered train index
        groups = pd.Series(cluster_groups, index=train.index).loc[train_c.index]
        model = smf.ols(formula=formula, data=train_c).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
    else:
        model = smf.ols(formula=formula, data=train_c).fit(cov_type=cov_type)
    pred = np.asarray(model.predict(test_c), dtype=float)
    return model, pred, train_c, test_c, means


def conformal_q(abs_residuals: Sequence[float], alpha: float = 0.05) -> float:
    r = np.sort(np.asarray(abs_residuals, dtype=float))
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return np.nan
    rank = int(math.ceil((n + 1) * (1 - alpha)))
    rank = min(max(rank, 1), n)
    return float(r[rank - 1])


def onehot_encoder_dense():
    """Compatibility helper across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def save_json(obj: Dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def ensure_outdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
