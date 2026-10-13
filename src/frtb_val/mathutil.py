"""Numerical helpers implemented with numpy and the standard library only."""
from __future__ import annotations

import math

import numpy as np

SQRT2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(x):
    """Standard normal CDF via erf; accepts scalars or arrays."""
    if np.isscalar(x):
        return 0.5 * (1.0 + math.erf(x / SQRT2))
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(np.asarray(x, dtype=float) / SQRT2))


def norm_pdf(x):
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / SQRT_2PI if not np.isscalar(x) \
        else math.exp(-0.5 * x * x) / SQRT_2PI


def norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam rational approximation plus one Halley step)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p <= 1 - plow:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    e = norm_cdf(x) - p
    u = e * SQRT_2PI * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    return min(1.0, sum(binom_pmf(i, n, p) for i in range(0, min(k, n) + 1)))


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k)."""
    return max(0.0, 1.0 - binom_cdf(k - 1, n, p))


def rankdata(x) -> np.ndarray:
    """Ranks starting at 1, ties receive the average rank."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc, yc = x - x.mean(), y - y.mean()
    denom = math.sqrt(float(xc @ xc) * float(yc @ yc))
    return float(xc @ yc) / denom if denom > 0 else float("nan")


def spearman(x, y) -> float:
    """Spearman rank correlation: Pearson correlation of the rank series."""
    return pearson(rankdata(x), rankdata(y))


def ks_statistic(x, y) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: max |ECDF_x - ECDF_y| over pooled values."""
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))
    pooled = np.concatenate([x, y])
    fx = np.searchsorted(x, pooled, side="right") / len(x)
    fy = np.searchsorted(y, pooled, side="right") / len(y)
    return float(np.max(np.abs(fx - fy)))


def ks_pvalue(d: float, n1: int, n2: int) -> float:
    """Asymptotic two-sample KS p-value Q(lambda), lambda = sqrt(n1 n2 / (n1 + n2)) * D."""
    lam = math.sqrt(n1 * n2 / (n1 + n2)) * d
    if lam < 0.2:
        return 1.0
    total = 0.0
    for j in range(1, 101):
        total += (-1) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam)
    return float(min(1.0, max(0.0, 2.0 * total)))


def brentq(f, a: float, b: float, tol: float = 1e-12, maxiter: int = 200) -> float:
    """Bracketing root finder (bisection with secant acceleration)."""
    fa, fb = f(a), f(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        raise ValueError("root not bracketed")
    for _ in range(maxiter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol or (b - a) < tol:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def expected_shortfall(pnl, alpha: float = 0.975) -> float:
    """Historical-simulation ES of a P&L sample, reported as a positive loss.

    Averages the worst (1 - alpha) * n losses with a fractional weight on the boundary
    observation, so the estimator is continuous in alpha.
    """
    losses = np.sort(-np.asarray(pnl, dtype=float))[::-1]
    n = len(losses)
    k = (1.0 - alpha) * n
    if k <= 0:
        return float(losses[0])
    whole = int(math.floor(k + 1e-12))
    frac = k - whole
    total = float(losses[:whole].sum())
    if frac > 1e-12 and whole < n:
        total += frac * float(losses[whole])
    return total / k


def value_at_risk(pnl, alpha: float = 0.99) -> float:
    """Historical-simulation VaR (positive loss): empirical alpha-quantile of losses."""
    losses = np.sort(-np.asarray(pnl, dtype=float))
    n = len(losses)
    idx = min(n - 1, int(math.ceil(alpha * n)) - 1)
    return float(losses[max(idx, 0)])
