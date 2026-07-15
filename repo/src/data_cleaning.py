"""
Cleaning and stratified sampling for the synthetic American-call dataset.

Cleaning enforces financial-domain constraints and the American-call
no-arbitrage bounds max(S - X, 0) <= Amercall <= S, then applies conservative
symmetric winsorisation. Sampling oversamples the interior of the state space
and under-samples the economically important tails, crossing moneyness and
maturity quantile bins into a maximum of 25 strata.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CORE_COLS = ["S", "X", "T", "r", "div", "v", "n", "Amercall"]


def clean_synthetic_data(raw_df: pd.DataFrame, trim_quantiles=(0.005, 0.995)):
    """Clean a raw synthetic American-call frame.

    Returns
    -------
    df_clean : pandas.DataFrame
        Cleaned rows, index reset.
    cleaning_audit : pandas.DataFrame
        Row counts removed at each step, for the reproducibility trail.
    retention_rate : float
        Fraction of the original rows retained.
    """
    df_clean = raw_df.copy()
    audit_rows = []

    def log_step(step_name, before, after):
        audit_rows.append(
            {
                "step": step_name,
                "rows_before": int(before),
                "rows_after": int(after),
                "rows_removed": int(before - after),
            }
        )

    start_n = len(df_clean)

    for col in CORE_COLS:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    log_step("drop_duplicates", before, len(df_clean))

    before = len(df_clean)
    df_clean = df_clean.dropna(subset=CORE_COLS)
    log_step("drop_missing_core_fields", before, len(df_clean))

    before = len(df_clean)
    df_clean = df_clean[
        (df_clean["S"] > 0)
        & (df_clean["X"] > 0)
        & (df_clean["T"] > 0)
        & (df_clean["v"] > 0)
        & (df_clean["r"] >= 0)
        & (df_clean["div"] >= 0)
        & (df_clean["Amercall"] >= 0)
    ].copy()
    log_step("financial_domain_constraints", before, len(df_clean))

    intrinsic = np.maximum(df_clean["S"] - df_clean["X"], 0)
    lower_bound = intrinsic
    upper_bound = df_clean["S"]

    before = len(df_clean)
    df_clean = df_clean[
        (df_clean["Amercall"] + 1e-10 >= lower_bound)
        & (df_clean["Amercall"] <= upper_bound + 1e-10)
    ].copy()
    log_step("no_arbitrage_bounds", before, len(df_clean))

    q_low, q_high = trim_quantiles
    trim_cols = ["S", "X", "v", "Amercall"]
    before = len(df_clean)
    for col in trim_cols:
        lo = df_clean[col].quantile(q_low)
        hi = df_clean[col].quantile(q_high)
        df_clean = df_clean[(df_clean[col] >= lo) & (df_clean[col] <= hi)]
    df_clean = df_clean.copy()
    log_step(f"quantile_trim_{q_low:.3f}_{q_high:.3f}", before, len(df_clean))

    cleaning_audit = pd.DataFrame(audit_rows)
    retention_rate = len(df_clean) / start_n if start_n else np.nan

    return df_clean.reset_index(drop=True), cleaning_audit, retention_rate


def stratified_sample(
    df: pd.DataFrame,
    n_bins: int = 5,
    min_per_stratum: int = 10,
    target_size: int = 4998,
    random_state: int = 42,
):
    """Stratified sample across moneyness and maturity quantile bins.

    Crosses ``n_bins`` moneyness bins with ``n_bins`` maturity bins to form up
    to ``n_bins**2`` strata, sampling proportionally with a floor per stratum.
    """
    work = df.copy()
    work["_m_bin"] = pd.qcut(work["S"] / work["X"], q=n_bins, duplicates="drop")
    work["_t_bin"] = pd.qcut(work["T"], q=n_bins, duplicates="drop")

    frac = min(1.0, target_size / len(work))
    sampled = []
    for _, group in work.groupby(["_m_bin", "_t_bin"], observed=True):
        take = max(min_per_stratum, int(round(len(group) * frac)))
        take = min(take, len(group))
        sampled.append(group.sample(n=take, random_state=random_state))

    out = pd.concat(sampled, ignore_index=True)
    return out.drop(columns=["_m_bin", "_t_bin"]).reset_index(drop=True)
