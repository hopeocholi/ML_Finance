"""
Closed-form option pricing used as a theory-based benchmark feature.

The Black-Scholes European call price enters the synthetic pipeline as an
explicit feature so that the machine-learning models learn the early-exercise
premium (the American-minus-European gap) rather than relearning closed-form
structure. Permutation importance on this feature then measures how much of
the American call value is captured in closed form.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bs_european_call_with_dividend(S, K, T, r, q, sigma):
    """Black-Scholes price for a dividend-paying European call.

    Vectorised over any broadcastable combination of inputs. Small floors are
    applied to time-to-maturity and volatility so that degenerate rows do not
    produce divide-by-zero or log-of-zero errors during feature engineering.

    Parameters
    ----------
    S : array_like
        Spot price of the underlying.
    K : array_like
        Strike price.
    T : array_like
        Time to maturity in years.
    r : array_like
        Continuously compounded risk-free rate.
    q : array_like
        Continuous dividend yield.
    sigma : array_like
        Volatility of the underlying.

    Returns
    -------
    numpy.ndarray
        European call price under the given dividend yield.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    eps = 1e-12
    T_safe = np.maximum(T, eps)
    sigma_safe = np.maximum(sigma, eps)

    d1 = (
        np.log(np.maximum(S, eps) / np.maximum(K, eps))
        + (r - q + 0.5 * sigma_safe ** 2) * T_safe
    ) / (sigma_safe * np.sqrt(T_safe))
    d2 = d1 - sigma_safe * np.sqrt(T_safe)

    return (
        np.exp(-q * T_safe) * S * norm.cdf(d1)
        - np.exp(-r * T_safe) * K * norm.cdf(d2)
    )
