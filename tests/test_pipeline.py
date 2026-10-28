"""Unit tests for the core pipeline components."""

import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import clean_synthetic_data, stratified_sample
from src.evaluation import paired_error_test, rmse, score_model
from src.features import engineer_synthetic_features
from src.pricing import bs_european_call_with_dividend


def test_bs_atm_reference_value():
    """ATM 1y call, r=5%, vol=20%, no dividend -> classic 10.4506."""
    price = bs_european_call_with_dividend(100, 100, 1.0, 0.05, 0.0, 0.20)
    assert float(price) == pytest.approx(10.4506, abs=1e-3)


def test_bs_respects_intrinsic_lower_bound():
    """Deep ITM call price must exceed intrinsic value S - K discounted."""
    price = float(bs_european_call_with_dividend(150, 100, 1.0, 0.05, 0.0, 0.20))
    assert price > 45.0


def test_bs_is_vectorised():
    S = np.array([90, 100, 110], dtype=float)
    out = bs_european_call_with_dividend(S, 100, 1.0, 0.05, 0.0, 0.2)
    assert out.shape == (3,)
    assert np.all(np.diff(out) > 0)  # increasing in spot


def _toy_synthetic_frame(n=500, seed=0):
    rng = np.random.default_rng(seed)
    S = rng.uniform(70, 130, n)
    X = np.full(n, 100.0)
    T = rng.uniform(0.1, 5.0, n)
    r = rng.uniform(0.0, 0.1, n)
    div = rng.uniform(0.0, 0.1, n)
    v = rng.uniform(0.1, 0.6, n)
    bs = bs_european_call_with_dividend(S, X, T, r, div, v)
    amercall = np.maximum(bs, np.maximum(S - X, 0)) + rng.uniform(0, 1, n)
    return pd.DataFrame(
        {"S": S, "X": X, "T": T, "r": r, "div": div, "v": v,
         "n": 15000, "Amercall": amercall}
    )


def test_cleaning_enforces_no_arbitrage():
    df = _toy_synthetic_frame()
    df.loc[0, "Amercall"] = -5.0  # violates lower bound
    clean, audit, retention = clean_synthetic_data(df)
    intrinsic = np.maximum(clean["S"] - clean["X"], 0)
    assert (clean["Amercall"] + 1e-9 >= intrinsic).all()
    assert (clean["Amercall"] <= clean["S"] + 1e-9).all()
    assert 0 < retention <= 1
    assert not audit.empty


def test_feature_engineering_adds_expected_columns():
    df = engineer_synthetic_features(_toy_synthetic_frame())
    for col in ["moneyness", "log_moneyness", "bs_euro_call", "vol_time"]:
        assert col in df.columns
    assert (df["early_exercise_proxy"] >= 0).all()


def test_stratified_sample_shrinks_and_preserves_columns():
    df = engineer_synthetic_features(_toy_synthetic_frame(n=2000))
    sampled = stratified_sample(df, target_size=400)
    assert len(sampled) <= len(df)
    assert set(["S", "X", "Amercall"]).issubset(sampled.columns)


def test_score_and_paired_test():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    good = y + 0.01
    bad = y + 0.5
    scores = score_model(y, good)
    assert scores["R2"] > 0.99
    result = paired_error_test(y, good, bad)
    assert result["mae_reduction"] > 0  # good beats bad


def test_rmse_matches_manual():
    y = np.array([0.0, 0.0, 0.0])
    pred = np.array([1.0, 1.0, 1.0])
    assert rmse(y, pred) == pytest.approx(1.0)
