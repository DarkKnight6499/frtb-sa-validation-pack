"""Outcomes analysis: model prices against a synthetic smile market, and a delta-hedging backtest with limits."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..mathutil import norm_cdf
from ..pricing import bs_implied_vol, bs_price
from .conceptual import MATURITIES, SPOT, STRIKES
from .model import FrontOfficePricer

# Limits are policy choices for this exercise, stated here so that tests are reproducible.
VOL_ERROR_LIMIT = 1.0          # absolute implied-vol error in vol points, for options within the moneyness band
MONEYNESS_BAND = 0.10          # |ln(K/F)| band inside which the limit applies
BREACH_RATE_LIMIT = 0.05       # share of in-band quotes allowed to breach
HEDGE_STD_LIMIT = 0.10         # std of terminal hedging error as a share of premium


def market_vol(k: float, t: float, fwd: float) -> float:
    """Synthetic smile: skew and curvature in log-moneyness plus a mild term structure. Not market data."""
    m = math.log(k / fwd)
    return 0.20 + 0.02 * math.exp(-t) - 0.15 * m + 0.40 * m * m


def pricing_outcomes(model: FrontOfficePricer, seed: int = 31, noise_vol: float = 0.002) -> pd.DataFrame:
    """Compare flat-vol model prices with prices generated from the synthetic smile (plus quote noise)."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in STRIKES:
        for t in MATURITIES:
            fwd = SPOT * math.exp((model.r - model.q) * t)
            vol = market_vol(k, t, fwd) + rng.normal(0, noise_vol)
            call = k >= fwd                                    # quote the out-of-the-money option
            mkt = bs_price(SPOT, k, model.r, model.q, vol, t, call)
            mod = model.price(SPOT, k, t, call)
            iv_model = bs_implied_vol(mod, SPOT, k, model.r, model.q, t, call)
            m = math.log(k / fwd)
            rows.append({"strike": k, "maturity": t, "log_moneyness": m, "type": "call" if call else "put",
                         "market_vol": vol, "model_vol": iv_model, "vol_error_pts": 100 * (iv_model - vol),
                         "market_price": mkt, "model_price": mod, "price_error": mod - mkt,
                         "in_band": abs(m) <= MONEYNESS_BAND})
    df = pd.DataFrame(rows)
    df["breach"] = df["in_band"] & (df["vol_error_pts"].abs() > VOL_ERROR_LIMIT)
    return df


def pricing_summary(df: pd.DataFrame) -> dict:
    band = df[df["in_band"]]
    return {"n_quotes": len(df), "n_in_band": len(band), "rmse_vol_pts": float(np.sqrt((df["vol_error_pts"] ** 2).mean())),
            "rmse_vol_pts_in_band": float(np.sqrt((band["vol_error_pts"] ** 2).mean())) if len(band) else float("nan"),
            "max_abs_vol_error_pts": float(df["vol_error_pts"].abs().max()),
            "breaches": int(band["breach"].sum()), "breach_rate": float(band["breach"].mean()) if len(band) else 0.0,
            "limit_passed": bool(len(band) == 0 or band["breach"].mean() <= BREACH_RATE_LIMIT)}


def _phi(x: np.ndarray) -> np.ndarray:
    return np.asarray(norm_cdf(x))


def hedging_backtest(model: FrontOfficePricer, true_vol: float | None = None, n_paths: int = 2000, n_steps: int = 250,
                     rebalance_every: int = 1, k: float = 100.0, t: float = 1.0, seed: int = 8) -> dict:
    """Sell a European call at model price and delta hedge at model vol; paths follow GBM at true_vol.

    Returns the distribution of terminal hedge error (replication shortfall) in absolute terms and per unit premium.
    """
    true_vol = model.sigma if true_vol is None else true_vol
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    r, q, sg = model.r, model.q, model.sigma
    s = np.full(n_paths, SPOT)
    premium = bs_price(SPOT, k, r, q, sg, t, True)
    cash = np.full(n_paths, premium)
    delta = np.zeros(n_paths)
    for i in range(n_steps):
        tau = t - i * dt
        if i % rebalance_every == 0:
            d1 = (np.log(s / k) + (r - q + 0.5 * sg ** 2) * tau) / (sg * math.sqrt(tau))
            new_delta = math.exp(-q * tau) * _phi(d1)
            cash -= (new_delta - delta) * s
            delta = new_delta
        cash *= math.exp(r * dt)
        cash += delta * s * (math.exp(q * dt) - 1.0)           # dividend yield earned on the hedge
        s = s * np.exp((r - q - 0.5 * true_vol ** 2) * dt + true_vol * math.sqrt(dt) * rng.standard_normal(n_paths))
    err = cash + delta * s - np.maximum(s - k, 0.0)
    return {"premium": premium, "mean_error": float(err.mean()), "std_error": float(err.std(ddof=1)),
            "std_over_premium": float(err.std(ddof=1) / premium), "p01_error": float(np.quantile(err, 0.01)),
            "p99_error": float(np.quantile(err, 0.99)), "true_vol": true_vol, "n_paths": n_paths, "rebalance_every": rebalance_every,
            "limit_passed": bool(err.std(ddof=1) / premium <= HEDGE_STD_LIMIT)}
