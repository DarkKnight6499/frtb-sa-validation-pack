"""The front-office model under validation: Black-Scholes (flat vol) for Europeans, CRR tree for American puts."""
from __future__ import annotations

from dataclasses import dataclass

from ..pricing import bs_greeks, bs_price, crr_binomial, crr_greeks


@dataclass(frozen=True)
class FrontOfficePricer:
    """Single flat volatility, constant rate and continuous dividend yield (the documented model assumptions)."""

    r: float = 0.05
    q: float = 0.02
    sigma: float = 0.20
    tree_steps: int = 500

    def price(self, s: float, k: float, t: float, call: bool, american: bool = False) -> float:
        if american:
            return crr_binomial(s, k, self.r, self.q, self.sigma, t, self.tree_steps, call, True)
        return bs_price(s, k, self.r, self.q, self.sigma, t, call)

    def greeks(self, s: float, k: float, t: float, call: bool, american: bool = False) -> dict:
        if american:
            return crr_greeks(s, k, self.r, self.q, self.sigma, t, self.tree_steps, call, True)
        return bs_greeks(s, k, self.r, self.q, self.sigma, t, call)
