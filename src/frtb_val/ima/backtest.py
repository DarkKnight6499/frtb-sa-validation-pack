"""VaR backtesting: exceptions, binomial tests, traffic light zones (MAR32.4 to MAR32.19, MAR99)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..mathutil import binom_cdf, binom_sf, value_at_risk

# MAR32.9 Table 1: backtesting multiplier add-on by exception count for 250 observations at 99%.
PLUS_FACTOR = {5: 0.20, 6: 0.26, 7: 0.33, 8: 0.38, 9: 0.42}     # multiplier minus the 1.5 base
BASE_MULTIPLIER = 1.5
AMBER_CUM = 0.95        # MAR99.17
RED_CUM = 0.9999
DESK_MAX_EXCEPTIONS = {0.99: 12, 0.975: 30}   # MAR32.19


def zone_boundaries(n: int = 250, coverage: float = 0.99) -> tuple[int, int]:
    """First exception counts of the amber and red zones from binomial cumulative probabilities (MAR99.17)."""
    p = 1.0 - coverage
    amber = next(k for k in range(n + 1) if binom_cdf(k, n, p) >= AMBER_CUM)
    red = next(k for k in range(n + 1) if binom_cdf(k, n, p) >= RED_CUM)
    return amber, red


def traffic_light(exceptions: int, n: int = 250, coverage: float = 0.99) -> str:
    amber, red = zone_boundaries(n, coverage)
    return "green" if exceptions < amber else ("amber" if exceptions < red else "red")


def multiplier(exceptions: int) -> float:
    """MAR32.9 Table 1 (250 observations, 99% VaR)."""
    if exceptions >= 10:
        return 2.0
    return BASE_MULTIPLIER + PLUS_FACTOR.get(exceptions, 0.0)


def binomial_exceedance_test(exceptions: int, n: int, coverage: float) -> dict:
    """One-sided binomial test that the exception rate is not above 1 - coverage, plus the lower-tail p-value."""
    p = 1.0 - coverage
    return {"expected": n * p, "p_too_many": binom_sf(exceptions, n, p), "p_too_few": binom_cdf(exceptions, n, p)}


def kupiec_pof(exceptions: int, n: int, coverage: float) -> dict:
    """Kupiec proportion-of-failures likelihood ratio, chi-square(1) p-value via erfc."""
    p = 1.0 - coverage
    x = exceptions
    phat = x / n

    def ll(q: float) -> float:
        return (n - x) * math.log(1 - q) + (x * math.log(q) if x else 0.0)

    lr = -2.0 * (ll(p) - ll(phat)) if 0 < phat < 1 else -2.0 * (n * math.log(1 - p) if x == 0 else 0.0)
    return {"lr": lr, "p_value": math.erfc(math.sqrt(max(lr, 0.0) / 2.0))}


@dataclass
class BacktestResult:
    alpha: float
    n: int
    exceptions: int
    zone: str
    p_too_many: float
    expected: float
    var_series: np.ndarray


def rolling_var(pnl: np.ndarray, alpha: float, window: int = 250, scale: float = 1.0) -> np.ndarray:
    """One-day HS VaR forecast for each day t from the previous `window` P&L observations (NaN before)."""
    out = np.full(len(pnl), np.nan)
    for t in range(window, len(pnl)):
        out[t] = scale * value_at_risk(pnl[t - window:t], alpha)
    return out


def backtest(pnl: np.ndarray, alpha: float, n: int = 250, window: int = 250, scale: float = 1.0) -> BacktestResult:
    """Count days in the last n where the loss exceeds the VaR forecast made from the preceding window."""
    pnl = np.asarray(pnl, dtype=float)
    var = rolling_var(pnl, alpha, window, scale)
    last = slice(len(pnl) - n, len(pnl))
    exc = int(np.sum(-pnl[last] > var[last]))
    cov = alpha
    zone = traffic_light(exc, n, cov) if alpha == 0.99 else _zone_other(exc, n, cov)
    test = binomial_exceedance_test(exc, n, cov)
    return BacktestResult(alpha, n, exc, zone, test["p_too_many"], test["expected"], var[last])


def _zone_other(exc: int, n: int, coverage: float) -> str:
    """Same binomial zone logic applied to other coverage levels (MAR32.13 permits other confidence levels)."""
    return traffic_light(exc, n, coverage)


def desk_backtest_fail(exceptions_99: int, exceptions_975: int) -> bool:
    """MAR32.19: more than 12 exceptions at 99% or 30 at 97.5% sends the desk to the standardised approach."""
    return exceptions_99 > DESK_MAX_EXCEPTIONS[0.99] or exceptions_975 > DESK_MAX_EXCEPTIONS[0.975]
