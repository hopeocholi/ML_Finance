# Methodology

This note records the protocol in enough detail to reproduce every number in the paper. Reporting decisions are fixed before any model is fitted, which is the main guard against train/test leakage.

## 1. Synthetic American-call track

### Data generation and cleaning
The dataset is a 15,000-step binomial American-call grid over a Monte Carlo design spanning spot in [70, 130], maturity in [0.1, 5.0] years, risk-free rate in [0, 0.1], dividend yield in [0, 0.1] and volatility in [0.1, 0.6], with strike fixed at 100. From 29,671 raw rows, cleaning removes:

1. duplicate rows;
2. rows with missing core fields;
3. rows violating financial-domain constraints (non-positive spot, strike, maturity or volatility);
4. rows violating the American-call no-arbitrage bounds, `max(S - X, 0) <= Amercall <= S`;
5. the outer 0.5% tails of scale-sensitive variables (symmetric winsorisation).

A positive continuous dividend yield keeps the early-exercise premium strictly positive, so the American-versus-European gap is non-degenerate and the regression target is economically meaningful.

### Feature engineering
Nine theory-based transformations augment the raw state variables: moneyness and log-moneyness, intrinsic value (a no-arbitrage bound), carry `r - q`, diffusion terms `sqrt(T)` and `v*sqrt(T)`, discount factors `exp(-rT)` and `exp(-qT)`, and the closed-form Black-Scholes European call price. The 14-dimensional feature vector drops the two constants (strike and step count).

### Sampling
The interior of the state space is oversampled and the economically important tails under-sampled. Five moneyness quantile bins are crossed with five maturity quantile bins to form up to 25 strata, sampled proportionally with a floor of 10 observations per stratum, yielding 4,998 contracts.

### Models and tuning
Six regressors share the same preprocessing (median imputation, z-score standardisation): Linear, Ridge (alpha = 1.0), Random Forest, Gradient Boosting, SVR-RBF and a two-hidden-layer MLP. A targeted `GridSearchCV` on Random Forest evaluates 24 candidates over 5 folds with negative-RMSE scoring.

## 2. Empirical SPY track

An SPY option chain is downloaded across all listed expiries with `yfinance`. Cleaning removes zero-bid, zero-ask, negative-spread and sub-one-day rows, and conservatively trims the outer 1% of spread, relative spread and implied volatility. The same six regressors are refitted twice: once with implied volatility as the target (13 predictors) and once with absolute bid-ask spread as the target (implied volatility substituted for mid-price).

## 3. Evaluation

- **Split.** 80/20 hold-out on the sampled data.
- **Cross-validation.** 5-fold `KFold`, shuffled, fixed seed, to test the stability of the hold-out ranking.
- **Metrics.** MAE (tail-resistant, in price units), RMSE and MSE (severe on large errors), and R².
- **Interpretability.** Permutation importance (MAE scoring, 10 repeats) on the best hold-out model.
- **Significance.** A paired t-test on the absolute errors of the top two models.

## Reproducibility

A single random seed (42) is set for NumPy and every estimator. The notebook exports its sampled synthetic dataset and empirical snapshot to CSV so the analysis can be re-run offline without hitting the network.
