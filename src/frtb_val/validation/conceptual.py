"""Conceptual soundness checks on the front-office pricer (SR 26-2 section V, conceptual soundness)."""
from __future__ import annotations

import math

import numpy as np

from ..pricing import bs_greeks, bs_implied_vol, bs_price, crr_binomial
from .model import FrontOfficePricer

STRIKES = (80.0, 90.0, 100.0, 110.0, 120.0)
MATURITIES = (0.25, 0.5, 1.0, 2.0)
SPOT = 100.0


def _record(name: str, passed: bool, detail: str) -> dict:
    return {"check": name, "passed": bool(passed), "detail": detail}


def run_conceptual_checks(model: FrontOfficePricer) -> list[dict]:
    """Return a list of {check, passed, detail}; every check is a numeric property computed on a grid."""
    r, q, sg = model.r, model.q, model.sigma
    out = []

    parity = max(abs(model.price(SPOT, k, t, True) - model.price(SPOT, k, t, False)
                     - (SPOT * math.exp(-q * t) - k * math.exp(-r * t))) for k in STRIKES for t in MATURITIES)
    out.append(_record("European put-call parity", parity < 1e-10, f"max residual {parity:.2e}"))

    viol = 0
    for k in STRIKES:
        for t in MATURITIES:
            c, p = model.price(SPOT, k, t, True), model.price(SPOT, k, t, False)
            lo_c = max(SPOT * math.exp(-q * t) - k * math.exp(-r * t), 0.0)
            lo_p = max(k * math.exp(-r * t) - SPOT * math.exp(-q * t), 0.0)
            viol += not (lo_c - 1e-12 <= c <= SPOT * math.exp(-q * t) + 1e-12)
            viol += not (lo_p - 1e-12 <= p <= k * math.exp(-r * t) + 1e-12)
    out.append(_record("No-arbitrage price bounds", viol == 0, f"{viol} violations on {2 * len(STRIKES) * len(MATURITIES)} prices"))

    spots = np.linspace(60, 140, 41)
    calls = [model.price(s, 100, 1.0, True) for s in spots]
    puts = [model.price(s, 100, 1.0, False) for s in spots]
    out.append(_record("Monotonicity in spot (call up, put down)",
                       all(np.diff(calls) > 0) and all(np.diff(puts) < 0), "41 spot points"))
    vols = np.linspace(0.05, 0.8, 30)
    vp = [bs_price(SPOT, 100, r, q, v, 1.0, True) for v in vols]
    out.append(_record("Monotonicity in volatility", all(np.diff(vp) > 0), "30 vol points"))
    strikes = np.linspace(60, 140, 41)
    cp = np.array([model.price(SPOT, k, 1.0, True) for k in strikes])
    out.append(_record("Convexity in strike (butterfly >= 0)", bool(np.all(cp[:-2] - 2 * cp[1:-1] + cp[2:] >= -1e-12)),
                       "41 strike points"))

    deltas_ok = gammas_ok = vegas_ok = True
    for k in STRIKES:
        for t in MATURITIES:
            gc, gp = bs_greeks(SPOT, k, r, q, sg, t, True), bs_greeks(SPOT, k, r, q, sg, t, False)
            deltas_ok &= 0 <= gc["delta"] <= 1 and -1 <= gp["delta"] <= 0
            gammas_ok &= gc["gamma"] >= 0 and abs(gc["gamma"] - gp["gamma"]) < 1e-12
            vegas_ok &= gc["vega"] >= 0 and abs(gc["vega"] - gp["vega"]) < 1e-9
    out.append(_record("Greek bounds (delta range, gamma and vega non-negative, call/put equalities)",
                       deltas_ok and gammas_ok and vegas_ok, "20 grid points"))

    h = 0.01
    fd_err = max(abs((bs_price(SPOT + h, k, r, q, sg, t, True) - bs_price(SPOT - h, k, r, q, sg, t, True)) / (2 * h)
                     - bs_greeks(SPOT, k, r, q, sg, t, True)["delta"]) for k in STRIKES for t in MATURITIES)
    out.append(_record("Analytic delta equals finite difference of price", fd_err < 1e-6, f"max error {fd_err:.2e}"))

    iv_err = max(abs(bs_implied_vol(bs_price(SPOT, k, r, q, v, t, True), SPOT, k, r, q, t, True) - v)
                 for k in STRIKES for t in MATURITIES for v in (0.1, 0.3, 0.6))
    out.append(_record("Implied volatility round trip", iv_err < 1e-8, f"max error {iv_err:.2e}"))

    t_small = 1e-6
    intr = max(SPOT * math.exp(-q * t_small) - 100 * math.exp(-r * t_small), 0.0)
    lim = abs(bs_price(SPOT, 100, r, q, sg, t_small, True) - intr)
    out.append(_record("Expiry limit: price tends to intrinsic as t tends to 0", lim < 1e-2, f"gap at t=1e-6 is {lim:.2e}"))

    gap = []
    ok = True
    for k in STRIKES:
        for t in MATURITIES:
            a = model.price(SPOT, k, t, False, american=True)
            e = crr_binomial(SPOT, k, r, q, sg, t, model.tree_steps, False, False)   # same tree, no early exercise
            ok &= a >= e - 1e-12 and a >= max(k - SPOT, 0.0) - 1e-12
            gap.append(a - e)
    out.append(_record("American put >= European put (same tree) and >= intrinsic", ok,
                       f"early-exercise premium from {min(gap):.4f} to {max(gap):.4f}"))

    diff = max(abs(crr_binomial(SPOT, k, r, 0.0, sg, t, 800, True, True) - crr_binomial(SPOT, k, r, 0.0, sg, t, 800, True, False))
               for k in STRIKES for t in MATURITIES)
    out.append(_record("American call equals European call when q = 0", diff < 1e-9, f"max difference {diff:.2e}"))

    bad = [n for n in (5, 10, 50, 500) for t in (0.01, 2.0)
           if not 0 < (math.exp((r - q) * t / n) - math.exp(-sg * math.sqrt(t / n))) / (math.exp(sg * math.sqrt(t / n)) - math.exp(-sg * math.sqrt(t / n))) < 1]
    out.append(_record("Tree risk-neutral probability inside (0, 1) for tested steps and maturities", not bad,
                       f"{len(bad)} violations"))
    return out


def assumption_register() -> list[dict]:
    """Model assumptions with the evidence section that tests each one."""
    return [
        {"assumption": "Lognormal diffusion, constant volatility across strike and maturity",
         "assessment": "Limitation: contradicted by any skew or term structure; quantified in outcomes analysis"},
        {"assumption": "Constant risk-free rate and continuous dividend yield",
         "assessment": "Acceptable for short-dated vanilla options; discrete dividends not supported"},
        {"assumption": "European exercise by closed form; American put by CRR tree with early exercise on nodes",
         "assessment": "Reasonable; tree discretisation error measured against an independent PDE solution"},
        {"assumption": "Frictionless continuous hedging",
         "assessment": "Limitation: daily discrete hedging error measured in outcomes analysis"},
        {"assumption": "American calls are priced by the same tree",
         "assessment": "Not benchmarked against a challenger: the PDE challenger supports American puts only"},
    ]
