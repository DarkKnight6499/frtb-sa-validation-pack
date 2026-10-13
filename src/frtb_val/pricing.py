"""Vanilla option pricing: Black-Scholes-Merton with continuous dividend yield, and CRR binomial."""
from __future__ import annotations

import math

import numpy as np

from .mathutil import brentq, norm_cdf, norm_pdf


def _d1d2(s: float, k: float, r: float, q: float, sigma: float, t: float) -> tuple[float, float]:
    vol_sqrt_t = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(s: float, k: float, r: float, q: float, sigma: float, t: float, call: bool = True) -> float:
    """Black-Scholes-Merton price. Degenerate inputs (t or sigma zero) return discounted intrinsic."""
    if t <= 0 or sigma <= 0:
        fwd_intrinsic = s * math.exp(-q * max(t, 0.0)) - k * math.exp(-r * max(t, 0.0))
        return max(fwd_intrinsic, 0.0) if call else max(-fwd_intrinsic, 0.0)
    d1, d2 = _d1d2(s, k, r, q, sigma, t)
    if call:
        return s * math.exp(-q * t) * norm_cdf(d1) - k * math.exp(-r * t) * norm_cdf(d2)
    return k * math.exp(-r * t) * norm_cdf(-d2) - s * math.exp(-q * t) * norm_cdf(-d1)


def bs_greeks(s: float, k: float, r: float, q: float, sigma: float, t: float, call: bool = True) -> dict:
    """Analytic delta, gamma, vega (per unit vol), theta (per year) and rho (per unit rate)."""
    d1, d2 = _d1d2(s, k, r, q, sigma, t)
    sq = math.sqrt(t)
    disc_q, disc_r = math.exp(-q * t), math.exp(-r * t)
    pdf = norm_pdf(d1)
    delta = disc_q * norm_cdf(d1) if call else -disc_q * norm_cdf(-d1)
    gamma = disc_q * pdf / (s * sigma * sq)
    vega = s * disc_q * pdf * sq
    if call:
        theta = (-s * disc_q * pdf * sigma / (2 * sq) - r * k * disc_r * norm_cdf(d2) + q * s * disc_q * norm_cdf(d1))
        rho = k * t * disc_r * norm_cdf(d2)
    else:
        theta = (-s * disc_q * pdf * sigma / (2 * sq) + r * k * disc_r * norm_cdf(-d2) - q * s * disc_q * norm_cdf(-d1))
        rho = -k * t * disc_r * norm_cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def bs_implied_vol(price: float, s: float, k: float, r: float, q: float, t: float, call: bool = True) -> float:
    """Implied volatility by bracketing root search on [1e-4, 5]."""
    return brentq(lambda v: bs_price(s, k, r, q, v, t, call) - price, 1e-4, 5.0, tol=1e-12)


def crr_binomial(s: float, k: float, r: float, q: float, sigma: float, t: float, steps: int,
                 call: bool = True, american: bool = False) -> float:
    """Cox-Ross-Rubinstein binomial tree, vectorised over each time slice."""
    dt = t / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    j = np.arange(steps + 1)
    prices = s * u ** (steps - j) * d ** j
    values = np.maximum(prices - k, 0.0) if call else np.maximum(k - prices, 0.0)
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if american:
            prices = s * u ** (i - np.arange(i + 1)) * d ** np.arange(i + 1)
            intrinsic = np.maximum(prices - k, 0.0) if call else np.maximum(k - prices, 0.0)
            values = np.maximum(values, intrinsic)
    return float(values[0])


def crr_greeks(s: float, k: float, r: float, q: float, sigma: float, t: float, steps: int,
               call: bool = True, american: bool = False) -> dict:
    """Delta and gamma from the first two tree layers (price is the root value)."""
    dt = t / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    n = steps
    j = np.arange(n + 1)
    prices = s * u ** (n - j) * d ** j
    values = np.maximum(prices - k, 0.0) if call else np.maximum(k - prices, 0.0)
    layers = {}
    for i in range(n - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        pr = s * u ** (i - np.arange(i + 1)) * d ** np.arange(i + 1)
        if american:
            intrinsic = np.maximum(pr - k, 0.0) if call else np.maximum(k - pr, 0.0)
            values = np.maximum(values, intrinsic)
        if i <= 2:
            layers[i] = (pr.copy(), values.copy())
    (p1, v1), (p2, v2) = layers[1], layers[2]
    delta = (v1[0] - v1[1]) / (p1[0] - p1[1])
    d_up = (v2[0] - v2[1]) / (p2[0] - p2[1])
    d_dn = (v2[1] - v2[2]) / (p2[1] - p2[2])
    gamma = (d_up - d_dn) / (0.5 * (p2[0] - p2[2]))
    return {"delta": float(delta), "gamma": float(gamma)}
