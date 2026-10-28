"""
Empirical SPY option-chain track: download, cleaning and modelling.

Real quotes carry market-microstructure signal that closed-form models do not
capture: dealer inventory, order-flow imbalance and stale quotes move implied
volatility and bid-ask spreads. The same six regressors are refitted with IV
and then spread as targets; the linear-versus-non-linear gap measures the
curvature that a closed-form surface cannot represent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .evaluation import score_model

RANDOM_STATE = 42

IV_FEATURES = [
    "strike",
    "underlying_price",
    "moneyness",
    "log_moneyness",
    "days_to_expiration",
    "T_years",
    "sqrt_T",
    "mid_price",
    "spread",
    "relative_spread",
    "volume",
    "openInterest",
    "is_call",
]

SPREAD_FEATURES = [
    "strike",
    "underlying_price",
    "moneyness",
    "log_moneyness",
    "days_to_expiration",
    "T_years",
    "sqrt_T",
    "impliedVolatility",
    "mid_price",
    "relative_spread",
    "volume",
    "openInterest",
    "is_call",
]


def download_option_chain_all_expiries(ticker_symbol: str) -> pd.DataFrame:
    """Download calls and puts across all listed expiries for a ticker.

    Requires ``yfinance``. Network access is needed at call time; the notebook
    caches its snapshot to CSV so the analysis is reproducible offline.
    """
    import yfinance as yf

    ticker_obj = yf.Ticker(ticker_symbol)
    expiries = list(ticker_obj.options)

    all_rows = []
    now_utc = pd.Timestamp.utcnow().tz_localize(None)

    for expiry in expiries:
        try:
            chain = ticker_obj.option_chain(expiry)
            for option_type, opt_df in [("Call", chain.calls), ("Put", chain.puts)]:
                temp = opt_df.copy()
                temp["expiry"] = pd.to_datetime(expiry)
                temp["option_type"] = option_type
                temp["ticker"] = ticker_symbol
                temp["download_timestamp_utc"] = now_utc
                if "lastTradeDate" in temp.columns:
                    temp["lastTradeDate"] = pd.to_datetime(
                        temp["lastTradeDate"], errors="coerce"
                    ).dt.tz_localize(None)
                all_rows.append(temp)
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"Skipping expiry {expiry} due to error: {exc}")

    if not all_rows:
        raise ValueError("No option-chain data downloaded. Try a more liquid ticker.")

    out = pd.concat(all_rows, ignore_index=True)
    px_hist = ticker_obj.history(period="5d")["Close"].dropna()
    if px_hist.empty:
        raise ValueError("Unable to retrieve recent underlying close.")
    out["underlying_spot_downloaded"] = float(px_hist.iloc[-1])
    return out


def prepare_option_chain_features(
    option_df: pd.DataFrame, underlying_price=None, iv_upper_cap: float = 5.0
) -> pd.DataFrame:
    """Clean and engineer empirical option-chain features for modelling."""
    df_opt = option_df.copy()

    if underlying_price is None:
        if "underlying_spot_downloaded" in df_opt.columns:
            underlying_price = float(
                df_opt["underlying_spot_downloaded"].dropna().iloc[0]
            )
        else:
            raise ValueError("No underlying price information available.")

    required_cols = ["strike", "bid", "ask", "impliedVolatility", "expiry"]
    for col in required_cols:
        if col not in df_opt.columns:
            raise ValueError(f"Missing required option-chain column: {col}")

    numeric_cols = [
        "strike", "bid", "ask", "impliedVolatility",
        "lastPrice", "volume", "openInterest", "inTheMoney",
    ]
    for col in numeric_cols:
        if col in df_opt.columns:
            df_opt[col] = pd.to_numeric(df_opt[col], errors="coerce")

    df_opt = df_opt.dropna(
        subset=["strike", "bid", "ask", "impliedVolatility", "expiry"]
    ).copy()

    df_opt = df_opt[
        (df_opt["strike"] > 0)
        & (df_opt["bid"] >= 0)
        & (df_opt["ask"] >= 0)
        & (df_opt["ask"] >= df_opt["bid"])
        & (df_opt["impliedVolatility"] > 0)
        & (df_opt["impliedVolatility"] < iv_upper_cap)
    ].copy()

    for col in ["volume", "openInterest"]:
        if col not in df_opt.columns:
            df_opt[col] = 0
        df_opt[col] = df_opt[col].fillna(0)

    df_opt["underlying_price"] = float(underlying_price)
    df_opt["mid_price"] = (df_opt["bid"] + df_opt["ask"]) / 2
    df_opt["spread"] = df_opt["ask"] - df_opt["bid"]
    df_opt["relative_spread"] = np.where(
        df_opt["mid_price"] > 0, df_opt["spread"] / df_opt["mid_price"], np.nan
    )

    now = pd.Timestamp.utcnow().tz_localize(None).normalize()
    df_opt["days_to_expiration"] = (
        pd.to_datetime(df_opt["expiry"]) - now
    ).dt.days
    df_opt["T_years"] = df_opt["days_to_expiration"] / 365.25
    df_opt["sqrt_T"] = np.sqrt(np.clip(df_opt["T_years"], 1e-8, None))

    df_opt["moneyness"] = df_opt["underlying_price"] / df_opt["strike"]
    df_opt["log_moneyness"] = np.log(np.clip(df_opt["moneyness"], 1e-12, None))
    df_opt["is_call"] = (
        df_opt["option_type"].str.lower() == "call"
    ).astype(int)

    df_opt = df_opt[
        (df_opt["mid_price"] > 0)
        & (df_opt["spread"] >= 0)
        & (df_opt["T_years"] > 0)
    ].copy()

    for col in ["spread", "relative_spread", "impliedVolatility"]:
        lo = df_opt[col].quantile(0.01)
        hi = df_opt[col].quantile(0.99)
        df_opt = df_opt[(df_opt[col] >= lo) & (df_opt[col] <= hi)]

    return df_opt.reset_index(drop=True)


def _empirical_model_library():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, max_depth=14, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.05, max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "SVR (RBF)": SVR(C=10.0, epsilon=0.02, kernel="rbf", gamma="scale"),
        "Multi-Layer Perceptron": MLPRegressor(
            hidden_layer_sizes=(64, 32), alpha=1e-4, max_iter=250,
            early_stopping=True, random_state=RANDOM_STATE,
        ),
    }


def _fit_target(option_df, features, target):
    df_model = option_df.dropna(subset=features + [target]).copy()
    X = df_model[features]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                features,
            )
        ]
    )

    rows, pred_store = [], {}
    for name, est in _empirical_model_library().items():
        model = Pipeline([("pre", pre), ("model", est)])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        pred_store[name] = pred
        rows.append({"Model": name, **score_model(y_test, pred)})

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best_name = results.iloc[0]["Model"]
    best_pred = pred_store[best_name]
    diagnostic = pd.DataFrame(
        {
            "actual": y_test.values,
            "predicted": best_pred,
            "residual": y_test.values - best_pred,
            "abs_error": np.abs(y_test.values - best_pred),
        }
    )
    return results, diagnostic


def fit_empirical_iv_models(option_df):
    """Fit the six regressors with implied volatility as the target."""
    return _fit_target(option_df, IV_FEATURES, "impliedVolatility")


def fit_empirical_spread_models(option_df):
    """Fit the six regressors with absolute bid-ask spread as the target."""
    return _fit_target(option_df, SPREAD_FEATURES, "spread")
