# Results

Full result tables reproduced from the accompanying paper.

## Synthetic American call

### Hold-out performance (80/20 split, sorted by Test RMSE)

| Model | Test MAE | Test RMSE | Test MSE | Test R² | Train R² |
|---|---|---|---|---|---|
| Gradient Boosting | 0.2102 | 0.3232 | 0.1044 | 0.9992 | 0.9995 |
| Neural Network (MLP) | 0.2410 | 0.3572 | 0.1276 | 0.9990 | 0.9991 |
| SVR (RBF) | 0.0955 | 0.3572 | 0.1276 | 0.9990 | 0.9998 |
| Random Forest | 0.1886 | 0.4255 | 0.1810 | 0.9986 | 0.9994 |
| Linear Regression | 0.4136 | 0.6173 | 0.3810 | 0.9971 | 0.9963 |
| Ridge Regression | 0.4134 | 0.6200 | 0.3844 | 0.9970 | 0.9963 |

Gradient Boosting has the lowest Test RMSE (0.323), 47.7% below Linear Regression. SVR has the lowest MAE (0.0955).

### Five-fold cross-validated performance

| Model | CV MAE | MAE SD | CV RMSE | RMSE SD | CV R² | R² SD |
|---|---|---|---|---|---|---|
| SVR (RBF) | 0.0954 | 0.0019 | 0.2859 | 0.0414 | 0.99936 | 0.0002 |
| Gradient Boosting | 0.2216 | 0.0089 | 0.3852 | 0.0480 | 0.99884 | 0.0003 |
| Neural Network (MLP) | 0.2903 | 0.0168 | 0.4156 | 0.0264 | 0.99866 | 0.0002 |
| Random Forest | 0.2218 | 0.0223 | 0.5672 | 0.0990 | 0.99742 | 0.0009 |
| Linear Regression | 0.4256 | 0.0126 | 0.6810 | 0.0550 | 0.99639 | 0.0006 |
| Ridge Regression | 0.4261 | 0.0119 | 0.6829 | 0.0539 | 0.99637 | 0.0006 |

Cross-validation reverses the hold-out ranking on MAE: SVR is the most stable learner on this smooth surface.

### Permutation importance (top six features, MAE scoring, 10 repeats)

| Feature | Importance mean | Importance SD |
|---|---|---|
| bs_euro_call | 10.6741 | 0.2049 |
| intrinsic_value | 0.7838 | 0.0342 |
| log_moneyness | 0.2546 | 0.0104 |
| div_discount | 0.2505 | 0.0114 |
| S (spot) | 0.2353 | 0.0105 |
| moneyness | 0.1801 | 0.0068 |

The Black-Scholes European price carries roughly 13.6 times the next feature. A sensitivity run without it collapses the non-linear learners to RMSE approximately 0.45, still well below Linear without the feature (approximately 1.1), so the pipeline adds value in either direction.

## Empirical SPY microstructure

### Implied volatility prediction

| Model | MAE | RMSE | MSE | R² |
|---|---|---|---|---|
| SVR (RBF) | 0.0137 | 0.0226 | 0.00051 | 0.9815 |
| Multi-Layer Perceptron | 0.0144 | 0.0280 | 0.00078 | 0.9715 |
| Random Forest | 0.0105 | 0.0317 | 0.00101 | 0.9635 |
| Gradient Boosting | 0.0179 | 0.0348 | 0.00121 | 0.9559 |
| Linear Regression | 0.0690 | 0.1037 | 0.01076 | 0.6091 |
| Ridge Regression | 0.0690 | 0.1037 | 0.01076 | 0.6091 |

### Bid-ask spread prediction

| Model | MAE | RMSE | MSE | R² |
|---|---|---|---|---|
| Multi-Layer Perceptron | 0.0569 | 0.1134 | 0.01287 | 0.9950 |
| Random Forest | 0.0355 | 0.1200 | 0.01440 | 0.9944 |
| Gradient Boosting | 0.0760 | 0.1661 | 0.02760 | 0.9892 |
| SVR (RBF) | 0.1047 | 0.2155 | 0.04643 | 0.9819 |
| Linear Regression | 0.7742 | 0.9637 | 0.92871 | 0.6382 |
| Ridge Regression | 0.7741 | 0.9637 | 0.92879 | 0.6382 |

The division of labour is not coincidental: SVR-RBF suits the smooth single non-linearity of the volatility smile, while the MLP captures the many interacting non-linearities of the spread surface.
