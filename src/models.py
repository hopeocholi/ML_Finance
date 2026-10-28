"""
The six-model regressor battery and shared preprocessing.

Every model shares the same preprocessing (median imputation, z-score
standardisation) so that differences in error are attributable to the learner
and not to the pipeline. Two linear baselines bracket the non-linear learners;
Random Forest and Gradient Boosting cover tree ensembles; SVR-RBF covers smooth
low-dimensional non-linearity; and an MLP covers higher-order interactions.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

RANDOM_STATE = 42


def build_preprocessor(numeric_features):
    """Median-impute then z-score standardise the numeric feature block."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer([("num", numeric_transformer, numeric_features)])


def build_model_library(preprocessor):
    """Return the six-model dictionary, each wrapped with the preprocessor."""
    return {
        "Linear Regression": Pipeline(
            [("preprocess", preprocessor), ("model", LinearRegression())]
        ),
        "Ridge Regression": Pipeline(
            [("preprocess", preprocessor), ("model", Ridge(alpha=1.0))]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=80,
                        max_depth=16,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=180,
                        learning_rate=0.05,
                        max_depth=3,
                        subsample=0.90,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "SVR": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    SVR(C=10.0, epsilon=0.05, kernel="rbf", gamma="scale"),
                ),
            ]
        ),
        "Neural Network (MLP)": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        alpha=1e-4,
                        learning_rate_init=0.001,
                        max_iter=250,
                        early_stopping=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
