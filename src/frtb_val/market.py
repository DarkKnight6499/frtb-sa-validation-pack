"""Synthetic market state with bump-and-revalue support."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

# MAR21.8 GIRR tenors, MAR21.9 CSR tenors, MAR21.12-14 vega maturity tenors.
GIRR_TENORS = np.array([0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30], dtype=float)
CSR_TENORS = np.array([0.5, 1, 3, 5, 10], dtype=float)
VOL_TENORS = np.array([0.5, 1, 3, 5, 10], dtype=float)

REPORTING_CCY = "USD"


@dataclass(frozen=True)
class Market:
    """Zero curves, credit spread curves, spots, dividend yields and vol term structures."""

    girr: dict = field(default_factory=dict)      # ccy -> zero rates at GIRR_TENORS (cont. comp.)
    csr: dict = field(default_factory=dict)       # name -> par spreads at CSR_TENORS
    spot: dict = field(default_factory=dict)      # equity name or foreign ccy -> USD price
    div: dict = field(default_factory=dict)       # equity name -> dividend yield
    vol_eq: dict = field(default_factory=dict)    # equity name -> vols at VOL_TENORS
    vol_fx: dict = field(default_factory=dict)    # foreign ccy -> vols at VOL_TENORS
    recovery: float = 0.4

    def zero(self, ccy: str, t: float) -> float:
        return float(np.interp(t, GIRR_TENORS, self.girr[ccy]))

    def df(self, ccy: str, t: float) -> float:
        return float(np.exp(-self.zero(ccy, t) * t))

    def spread(self, name: str, t: float) -> float:
        return float(np.interp(t, CSR_TENORS, self.csr[name]))

    def vol(self, kind: str, key: str, t: float) -> float:
        table = self.vol_eq if kind == "vol_eq" else self.vol_fx
        return float(np.interp(t, VOL_TENORS, table[key]))

    def bumped(self, kind: str, key: str, idx: int | None, amount: float, relative: bool = False) -> "Market":
        """Return a copy with one risk factor shifted (absolute, or relative when requested)."""
        def shift(arr: np.ndarray) -> np.ndarray:
            out = arr.copy()
            out[idx] = out[idx] * (1 + amount) if relative else out[idx] + amount
            return out

        if kind == "girr":
            return replace(self, girr={**self.girr, key: shift(self.girr[key])})
        if kind == "csr":
            return replace(self, csr={**self.csr, key: shift(self.csr[key])})
        if kind in ("eq", "fx"):
            new = self.spot[key] * (1 + amount) if relative else self.spot[key] + amount
            return replace(self, spot={**self.spot, key: new})
        if kind == "vol_eq":
            return replace(self, vol_eq={**self.vol_eq, key: shift(self.vol_eq[key])})
        if kind == "vol_fx":
            return replace(self, vol_fx={**self.vol_fx, key: shift(self.vol_fx[key])})
        raise KeyError(kind)

    def bumped_parallel(self, kind: str, key: str, amount: float) -> "Market":
        """Parallel absolute shift of a whole curve (used for curvature scenarios)."""
        if kind == "girr":
            return replace(self, girr={**self.girr, key: self.girr[key] + amount})
        if kind == "csr":
            return replace(self, csr={**self.csr, key: self.csr[key] + amount})
        raise KeyError(kind)


def build_market(seed: int = 20260101) -> Market:
    """Seeded synthetic market. All levels are invented and carry no market information."""
    rng = np.random.default_rng(seed)
    base = {"USD": 0.040, "EUR": 0.025, "GBP": 0.043}
    girr = {}
    for ccy, level in base.items():
        slope = rng.uniform(0.002, 0.006)
        girr[ccy] = level + slope * (1 - np.exp(-GIRR_TENORS / 4.0)) + rng.normal(0, 0.0004, len(GIRR_TENORS))
    names = {
        "SOV_A": 0.0040, "FIN_A": 0.0090, "IND_A": 0.0110, "CNS_A": 0.0130,
        "TEC_A": 0.0100, "HC_A": 0.0120, "FIN_HY": 0.0420, "CNS_HY": 0.0480,
        "IG_INDEX": 0.0075,
    }
    csr = {n: lvl * (1 + 0.15 * np.log1p(CSR_TENORS)) * (1 + rng.normal(0, 0.02, len(CSR_TENORS)))
           for n, lvl in names.items()}
    spot = {"EQ_TECH": 185.0, "EQ_FIN": 92.0, "EQ_CONS": 64.0, "EQ_ENER": 118.0, "EQ_SMALL": 27.0,
            "EQ_INDEX": 4500.0, "EUR": 1.08, "GBP": 1.27}
    div = {"EQ_TECH": 0.006, "EQ_FIN": 0.025, "EQ_CONS": 0.020, "EQ_ENER": 0.035, "EQ_SMALL": 0.0,
           "EQ_INDEX": 0.015}
    vol_eq = {n: lvl + 0.02 * np.exp(-VOL_TENORS / 3.0) for n, lvl in
              {"EQ_TECH": 0.30, "EQ_FIN": 0.26, "EQ_CONS": 0.20, "EQ_ENER": 0.28, "EQ_SMALL": 0.40,
               "EQ_INDEX": 0.17}.items()}
    vol_fx = {n: lvl + 0.01 * np.exp(-VOL_TENORS / 2.0) for n, lvl in {"EUR": 0.075, "GBP": 0.085}.items()}
    return Market(girr=girr, csr=csr, spot=spot, div=div, vol_eq=vol_eq, vol_fx=vol_fx)
