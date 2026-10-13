"""Risk factor eligibility test (MAR31.12, MAR31.13) and a simplified NMRF stress scenario charge (MAR33.16, 33.17)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..mathutil import expected_shortfall
from .factors import FACTOR_ID, LIQ_HORIZON

# MAR31.13: verified against the January 2019 text. Criterion (1): >= 24 real price observations over the
# calibration year and no 90-day period in the previous 12 months with fewer than 4. Criterion (2): >= 100
# observations over the previous 12 months. At most one observation per day counts.
MIN_OBS_CRITERION_1 = 24
MIN_OBS_PER_90_DAYS = 4
MIN_OBS_CRITERION_2 = 100
YEAR_DAYS = 365
WINDOW_DAYS = 90
NMRF_MIN_HORIZON = 20                 # MAR33.16(1)
SES_RHO = 0.6                         # MAR33.17(4)


def rfet(obs_days) -> dict:
    """Apply the RFET to the calendar days (0..364) on which a real price observation was identified."""
    days = np.unique(np.asarray(list(obs_days), dtype=int))
    days = days[(days >= 0) & (days < YEAR_DAYS)]
    n = len(days)
    flags = np.zeros(YEAR_DAYS, dtype=int)
    flags[days] = 1
    csum = np.concatenate([[0], np.cumsum(flags)])
    window_counts = csum[WINDOW_DAYS:] - csum[:-WINDOW_DAYS]     # every 90-day window inside the year
    min_window = int(window_counts.min())
    crit1 = n >= MIN_OBS_CRITERION_1 and min_window >= MIN_OBS_PER_90_DAYS
    crit2 = n >= MIN_OBS_CRITERION_2
    return {"n_obs": n, "min_obs_in_90d": min_window, "criterion_1": bool(crit1), "criterion_2": bool(crit2),
            "modellable": bool(crit1 or crit2)}


def synthetic_observations(meta: pd.DataFrame, seed: int = 5) -> dict:
    """Seeded real-price observation days per factor. Liquidity depends on the factor's liquidity horizon.

    Factors with a long liquidity horizon trade rarely and some are clustered, so the 90-day gap rule can bind
    even when the annual count passes.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for _, row in meta.iterrows():
        lh = row[LIQ_HORIZON]
        rate = {10: 220, 20: 130, 40: 60, 60: 28, 120: 9}[int(lh)] * rng.uniform(0.7, 1.4)
        n = rng.poisson(rate)
        days = rng.integers(0, YEAR_DAYS, n)
        if lh == 60 and rng.uniform() < 0.5:           # clustered trading: a quiet stretch breaks the 90-day rule
            days = days % 200
        out[row[FACTOR_ID]] = days
    return out


def stress_scenario_charge(factor_changes: np.ndarray, s: float, g: float, lh: int, alpha: float = 0.975) -> float:
    """Single-factor stress scenario: ES of the P&L over max(LH, 20) days in the stress window (MAR33.16)."""
    h = max(int(lh), NMRF_MIN_HORIZON)
    c = np.concatenate([[0.0], np.cumsum(factor_changes)])
    x = c[h:] - c[:-h]
    return float(max(expected_shortfall(s * x + 0.5 * g * x ** 2, alpha), 0.0))


def ses(ses_k: np.ndarray, ises_i: np.ndarray | None = None, ises_j: np.ndarray | None = None,
        rho: float = SES_RHO) -> float:
    """MAR33.17: SES = sqrt(sum ISES_i^2) + sqrt(sum ISES_j^2) + sqrt((sum rho SES_k)^2 + sum (1-rho^2) SES_k^2)."""
    ises_i = np.zeros(0) if ises_i is None else np.asarray(ises_i)
    ises_j = np.zeros(0) if ises_j is None else np.asarray(ises_j)
    ses_k = np.asarray(ses_k)
    return float(math.sqrt((ises_i ** 2).sum()) + math.sqrt((ises_j ** 2).sum())
                 + math.sqrt((rho * ses_k.sum()) ** 2 + ((1 - rho ** 2) * ses_k ** 2).sum()))
