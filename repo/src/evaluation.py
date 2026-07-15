"""
Evaluation, cross-validation and interpretability utilities.

Reporting is fixed before any model is fitted: MAE (tail-resistant, in price
units), RMSE and MSE (severe on large errors), and R2. A 5-fold cross-validation
tests the stability of the hold-out ranking; permutation importance decomposes
the pricing error at the feature level, which the EU AI Act framing makes a
first-class requirement rather than an optional extra.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_validate

RANDOM_STATE = 42


def rmse(y_true, y_pred):
    """Root mean squared error in the units of the target."""
    return math.sqrt(mean_squared_error(y_true, y_pred))


def score_model(y_true, y_pred):
    """Return the four headline metrics as a dict."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Fit each model and return a hold-out results frame sorted by RMSE."""
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        rows.append(
            {
                "Model": name,
                "Test_MAE": mean_absolute_error(y_test, test_pred),
                "Test_RMSE": rmse(y_test, test_pred),
                "Test_MSE": mean_squared_error(y_test, test_pred),
                "Test_R2": r2_score(y_test, test_pred),
                "Train_R2": r2_score(y_train, train_pred),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("Test_RMSE")
        .reset_index(drop=True)
    )


def cross_validate_models(models, X, y, n_splits=5):
    """Five-fold CV means and standard deviations for MAE, RMSE and R2."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    rows = []
    for name, model in models.items():
        cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append(
            {
                "Model": name,
                "CV_MAE": -cv_res["test_mae"].mean(),
                "MAE_SD": cv_res["test_mae"].std(),
                "CV_RMSE": -cv_res["test_rmse"].mean(),
                "RMSE_SD": cv_res["test_rmse"].std(),
                "CV_R2": cv_res["test_r2"].mean(),
                "R2_SD": cv_res["test_r2"].std(),
            }
        )
    return pd.DataFrame(rows).sort_values("CV_MAE").reset_index(drop=True)


def compute_permutation_importance(
    fitted_model, X_test, y_test, feature_names, n_repeats=10
):
    """Permutation importance (MAE scoring) as a sorted frame."""
    result = permutation_importance(
        fitted_model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance_Mean": result.importances_mean,
                "Importance_SD": result.importances_std,
            }
        )
        .sort_values("Importance_Mean", ascending=False)
        .reset_index(drop=True)
    )


def paired_error_test(y_true, pred_a, pred_b):
    """Paired t-test on absolute errors of two models.

    Returns the t-statistic, p-value and the mean absolute-error reduction of
    model A relative to model B (positive means A is more accurate).
    """
    from scipy import stats

    abs_a = np.abs(y_true - pred_a)
    abs_b = np.abs(y_true - pred_b)
    t_stat, p_value = stats.ttest_rel(abs_a, abs_b)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "mae_reduction": float(abs_b.mean() - abs_a.mean()),
    }
