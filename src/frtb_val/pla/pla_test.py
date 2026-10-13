"""P&L attribution test (MAR32.34 to MAR32.44): Spearman correlation and Kolmogorov-Smirnov metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ima.backtest import desk_backtest_fail
from ..mathutil import ks_pvalue, ks_statistic, spearman

# MAR32.42 Table 2, verified against the January 2019 text.
SPEARMAN_GREEN = 0.80
SPEARMAN_RED = 0.70
KS_GREEN = 0.09
KS_RED = 0.12
PLA_OBSERVATIONS = 250   # MAR32.35


def pla_zone(corr: float, ks: float) -> str:
    """Green if corr > 0.80 and KS < 0.09; red if corr < 0.70 or KS > 0.12; otherwise amber (MAR32.42)."""
    if corr < SPEARMAN_RED or ks > KS_RED:
        return "red"
    if corr > SPEARMAN_GREEN and ks < KS_GREEN:
        return "green"
    return "amber"


@dataclass
class PLAResult:
    spearman: float
    ks: float
    ks_pvalue: float
    zone: str
    n: int


def pla_test(hpl, rtpl) -> PLAResult:
    """Run the test on the most recent 250 observations of hypothetical and risk-theoretical P&L."""
    hpl = np.asarray(hpl, dtype=float)[-PLA_OBSERVATIONS:]
    rtpl = np.asarray(rtpl, dtype=float)[-PLA_OBSERVATIONS:]
    if len(hpl) != len(rtpl):
        raise ValueError("series must have equal length")
    corr = spearman(hpl, rtpl)
    ks = ks_statistic(rtpl, hpl)
    return PLAResult(corr, ks, ks_pvalue(ks, len(hpl), len(rtpl)), pla_zone(corr, ks), len(hpl))


def desk_eligibility(pla_zone_name: str, exceptions_99: int, exceptions_975: int) -> str:
    """Desk-level outcome combining PLA zone (MAR32.43, 32.44) and backtesting (MAR32.19)."""
    if pla_zone_name == "red":
        return "STANDARDISED (PLA red)"
    if desk_backtest_fail(exceptions_99, exceptions_975):
        return "STANDARDISED (backtesting)"
    if pla_zone_name == "amber":
        return "IMA with capital surcharge (PLA amber)"
    return "IMA eligible"
