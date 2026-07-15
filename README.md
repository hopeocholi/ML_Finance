# Machine Learning for American Option Pricing and Option-Chain Microstructure

Benchmarking six supervised regressors against the Black-Scholes surface on a high-step binomial American-call dataset, and predicting implied volatility and bid-ask spreads on real SPY option chains.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![Tests](https://img.shields.io/badge/tests-pytest-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Author: **Hope Eneojo Ocholi**

---

## Overview

The project answers two research questions with a single, disciplined pipeline.

**RQ1.** Do machine-learning regressors outperform a Black-Scholes European model in estimating American call values, and is any outperformance due to the early-exercise premium or to the closed-form component?

**RQ2.** To what extent can machine-learning regressors predict implied volatility and bid-ask spreads on real SPY quotes, and does the linear-versus-non-linear gap reveal microstructure curvature that closed-form models cannot see?

The design principle throughout is that the Black-Scholes European call price enters as an explicit feature, so the models learn the residual (the early-exercise premium) rather than relearning closed-form structure. Permutation importance then decomposes the pricing error at the feature level.

## Headline results

**Synthetic American-call track (4,998 stratified contracts).** Gradient Boosting attains the lowest hold-out error, and a cross-validated SVR attains the lowest CV-MAE.

![Synthetic RMSE by model](figures/synthetic_rmse_by_model.png)

| Model | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| **Gradient Boosting** | 0.210 | **0.323** | 0.9992 |
| Neural Network (MLP) | 0.241 | 0.357 | 0.9990 |
| SVR (RBF) | **0.095** | 0.357 | 0.9990 |
| Random Forest | 0.189 | 0.425 | 0.9986 |
| Linear Regression | 0.414 | 0.617 | 0.9971 |
| Ridge Regression | 0.413 | 0.620 | 0.9970 |

Permutation importance shows the Black-Scholes European price carries roughly 13.6 times the explanatory power of the next feature. The reading is that the feature set converts the problem into residual pricing, and the learned residual is the early-exercise premium.

![Permutation importance](figures/permutation_importance.png)

**Empirical SPY track (8,063 cleaned quotes, 32 expiries).** Non-linear models capture the volatility smile and the discrete tick-size regimes that linear baselines collapse on.

![Empirical R2 comparison](figures/empirical_r2_comparison.png)

| Target | Best model | R² | Linear baseline R² |
|---|---|---|---|
| Implied volatility | SVR (RBF) | 0.981 | 0.609 |
| Bid-ask spread | MLP | 0.994 | 0.638 |

## Repository layout

```
.
├── src/                     # Reusable, tested pipeline package
│   ├── pricing.py           # Black-Scholes European call (dividend yield)
│   ├── features.py          # Theory-motivated feature engineering
│   ├── data_cleaning.py     # No-arbitrage cleaning + stratified sampling
│   ├── models.py            # Six-model regressor battery + preprocessing
│   ├── evaluation.py        # Metrics, cross-validation, permutation importance
│   └── option_chain.py      # SPY download, cleaning, IV + spread modelling
├── notebooks/               # End-to-end research notebook (the evidence layer)
├── paper/                   # IEEE-format manuscript (PDF)
├── figures/                 # Summary figures used in this README
├── tests/                   # pytest unit tests for the pipeline
├── data/                    # Placeholder for exported CSV artefacts
├── docs/                    # Methodology and results notes
├── requirements.txt
└── README.md

```

### Using the package directly

```python
from src.pricing import bs_european_call_with_dividend
from src.features import engineer_synthetic_features, SYNTHETIC_FEATURES
from src.models import build_preprocessor, build_model_library
from src.evaluation import evaluate_models, compute_permutation_importance

# Black-Scholes European call with a continuous dividend yield
price = bs_european_call_with_dividend(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20)
# -> 10.4506
```

## Methodology at a glance

- **Data.** A 15,000-step binomial American-call dataset (29,671 raw rows) is cleaned against no-arbitrage bounds and winsorised, then stratified by moneyness and maturity down to 4,998 contracts. The empirical track is an SPY snapshot from Yahoo Finance (8,722 raw, 8,063 after cleaning across 32 expiries).
- **Features.** Nine theory-based transformations encode no-arbitrage bounds, diffusion scaling, discounting and the closed-form Black-Scholes benchmark.
- **Models.** Linear, Ridge, Random Forest, Gradient Boosting, SVR-RBF and an MLP, all sharing median imputation and z-score standardisation.
- **Evaluation.** 80/20 hold-out plus 5-fold cross-validation; MAE, RMSE, MSE and R²; permutation importance; a paired t-test on the top two models' absolute errors.

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the full protocol and [`docs/RESULTS.md`](docs/RESULTS.md) for the extended result tables.

## Limitations

The synthetic track fixes the strike and step count and does not test stochastic strikes; the empirical snapshot is a single trading day rather than a panel; the risk-free rate is held constant; and the feature set excludes order-flow and ETF creation-redemption variables. These are set out in full in the paper's Section on robustness and limitations.

## Citation

If you reference this work, please cite the accompanying paper:

```bibtex
@techreport{ocholi2026mloption,
  title  = {Machine Learning for American Option Pricing and Option-Chain
            Microstructure: Benchmark Trading Against the Black-Scholes Surface
            on Synthetic and Real-World SPY Data},
  author = {Ocholi, Hope Eneojo},
  year   = {2026},
  institution = {National College of Ireland}
}
```

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
