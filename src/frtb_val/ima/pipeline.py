"""Glue: build factor table, synthetic panel and P&L contribution arrays from the synthetic book."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import columns as C
from ..portfolio import default_book_and_market, sensitivities
from . import factors as F

BASE_HORIZON = 10


@dataclass
class IMAInputs:
    sens: pd.DataFrame
    meta: pd.DataFrame
    panel: pd.DataFrame
    daily: np.ndarray        # daily P&L contributions, days x factors
    contrib10: np.ndarray    # overlapping 10-day P&L contributions, scenarios x factors
    lh: np.ndarray
    broad: np.ndarray


def build_inputs(book_seed: int = 7, panel_seed: int = 11, n_days: int = 2500, stress_start: int = 700) -> IMAInputs:
    book, market = default_book_and_market(book_seed)
    sens = sensitivities(book, market)
    meta = F.factor_table(sens)
    panel = F.simulate_panel(meta, n_days=n_days, stress_start=stress_start, seed=panel_seed)
    s, g = meta[C.SENSITIVITY].to_numpy(), meta[C.GAMMA].to_numpy()
    x = panel.to_numpy()
    daily = F.contributions(x, s, g)
    contrib10 = F.contributions(F.horizon_changes(x, BASE_HORIZON), s, g)
    return IMAInputs(sens, meta, panel, daily, contrib10, meta[F.LIQ_HORIZON].to_numpy(),
                     meta[F.BROAD_CLASS].to_numpy())
