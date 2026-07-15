"""
American option pricing and option-chain microstructure with machine learning.

A reproducible pipeline that (1) benchmarks six regressors against the
Black-Scholes surface on a high-step binomial American-call dataset, and
(2) predicts implied volatility and bid-ask spreads on real SPY option chains.

Modules
-------
pricing        Black-Scholes European call with continuous dividend yield.
features       Theory-motivated feature engineering for the synthetic track.
data_cleaning  No-arbitrage cleaning and stratified sampling.
models         The six-model regressor battery and shared preprocessing.
evaluation     MAE, RMSE, MSE, R2, cross-validation and permutation importance.
option_chain   Yahoo Finance download, cleaning, IV and spread modelling.
"""

__version__ = "1.0.0"
__author__ = "Hope Eneojo Ocholi"

from .pricing import bs_european_call_with_dividend
from .evaluation import rmse

__all__ = ["bs_european_call_with_dividend", "rmse", "__version__"]
