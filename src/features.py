"""
Theory-motivated feature engineering for the synthetic American-call track.

Nine transformations augment the raw state variables. They encode the
no-arbitrage lower bound, diffusion scaling, discounting structure and the
closed-form Black-Scholes European benchmark. With a positive dividend yield
the target retains a strictly positive early-exercise premium, so the learned
residual is economically meaningful rather than a rediscovery of the formula.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pricing import bs_european_call_with_dividend

# Feature vector consumed by the synthetic-track models. Constants X (strike,
# fixed at 100) and n (step count, fixed at 15,000) are intentionally dropped.
SYNTHETIC_FEATURES = [
    "S",
    "T",
    "r",
    "div",
    "v",
    "moneyness",
    "log_moneyness",
    "intrinsic_value",
    "carry",
    "sqrt_T",
    "vol_time",
    "discount_factor",
    "div_discount",
    "bs_euro_call",
]

TARGET_COL = "Amercall"


def engineer_synthetic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add theory-based features to a cleaned synthetic American-call frame.

    Expects the raw columns ``S, X, T, r, div, v`` and the target ``Amercall``.
    Returns a copy with the engineered columns appended.
    """
    df = df.copy()

    df["moneyness"] = df["S"] / df["X"]
    df["log_moneyness"] = np.log(df["moneyness"])
    df["intrinsic_value"] = np.maximum(df["S"] - df["X"], 0)
    df["carry"] = df["r"] - df["div"]
    df["sqrt_T"] = np.sqrt(df["T"])
    df["vol_time"] = df["v"] * df["sqrt_T"]
    df["discount_factor"] = np.exp(-df["r"] * df["T"])
    df["div_discount"] = np.exp(-df["div"] * df["T"])
    df["bs_euro_call"] = bs_european_call_with_dividend(
        df["S"], df["X"], df["T"], df["r"], df["div"], df["v"]
    )
    df["early_exercise_proxy"] = np.maximum(
        df["Amercall"] - df["bs_euro_call"], 0
    )

    return df
