"""Benchmark, convergence and Greeks comparisons of the front-office pricer against the challengers."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..pricing import bs_greeks, bs_price, crr_binomial, crr_greeks
from .challenger import fd_option, mc_european, mc_greeks_crn
from .conceptual import MATURITIES, SPOT, STRIKES
from .model import FrontOfficePricer

MC_PATHS = 200_000
FD_GRID = 800
Z_LIMIT = 3.0              # a Monte Carlo difference beyond 3 standard errors is a benchmark failure


def benchmark_grid(model: FrontOfficePricer, seed: int = 2024, mc_paths: int = MC_PATHS, fd_grid: int = FD_GRID) -> pd.DataFrame:
    """European call and put: closed form vs Monte Carlo with control variate (z-score) and vs finite difference."""
    rows = []
    for k in STRIKES:
        for t in MATURITIES:
            for call in (True, False):
                fo = model.price(SPOT, k, t, call)
                mc = mc_european(SPOT, k, model.r, model.q, model.sigma, t, call, mc_paths, seed + int(k) + int(t * 100))
                fd = fd_option(SPOT, k, model.r, model.q, model.sigma, t, call, False, fd_grid, fd_grid)["price"]
                rows.append({"type": "call" if call else "put", "strike": k, "maturity": t, "front_office": fo,
                             "mc_cv": mc.price, "mc_stderr": mc.stderr, "z_score": (fo - mc.price) / mc.stderr,
                             "fd": fd, "fo_minus_fd": fo - fd, "mc_ci_contains": mc.ci95[0] <= fo <= mc.ci95[1],
                             "variance_reduction": mc.variance_reduction})
    return pd.DataFrame(rows)


def american_benchmark(model: FrontOfficePricer, fd_grid: int = FD_GRID) -> pd.DataFrame:
    """American put: front-office tree vs Crank-Nicolson PDE with Brennan-Schwartz projection."""
    rows = []
    for k in STRIKES:
        for t in MATURITIES:
            fo = model.price(SPOT, k, t, False, american=True)
            fd = fd_option(SPOT, k, model.r, model.q, model.sigma, t, False, True, fd_grid, fd_grid)["price"]
            ref = crr_binomial(SPOT, k, model.r, model.q, model.sigma, t, 5000, False, True)
            eur = model.price(SPOT, k, t, False)
            rows.append({"strike": k, "maturity": t, "front_office_tree": fo, "fd_challenger": fd, "tree_5000": ref,
                         "tree_minus_fd": fo - fd, "tree_minus_ref": fo - ref, "early_exercise_premium": fo - eur})
    return pd.DataFrame(rows)


def convergence(model: FrontOfficePricer) -> dict:
    """Error against refinement for the binomial tree, Monte Carlo and the finite-difference grid."""
    s, k, t = SPOT, 100.0, 1.0
    exact = bs_price(s, k, model.r, model.q, model.sigma, t, False)
    steps = [10, 20, 40, 80, 160, 320, 640, 1280, 2560]
    tree = pd.DataFrame([{"steps": n, "price": (p := crr_binomial(s, k, model.r, model.q, model.sigma, t, n, False)),
                          "abs_error": abs(p - exact)} for n in steps])
    tree_order = float(np.polyfit(np.log(tree["steps"]), np.log(tree["abs_error"]), 1)[0])
    odd = [abs(crr_binomial(s, k, model.r, model.q, model.sigma, t, n, False) - exact) for n in (200, 201, 202, 203)]
    paths = [2 ** i for i in range(10, 19, 2)]
    mc_rows = []
    for n in paths:
        res = mc_european(s, k, model.r, model.q, model.sigma, t, False, n, seed=77)
        mc_rows.append({"paths": n, "price": res.price, "abs_error": abs(res.price - exact), "stderr_cv": res.stderr,
                        "stderr_plain": res.plain_stderr, "variance_reduction": res.variance_reduction})
    mc = pd.DataFrame(mc_rows)
    mc_order = float(np.polyfit(np.log(mc["paths"]), np.log(mc["stderr_cv"]), 1)[0])
    ref = crr_binomial(s, k, model.r, model.q, model.sigma, t, 8000, False, True)
    grids = [50, 100, 200, 400, 800, 1600]
    fd = pd.DataFrame([{"grid": g, "price": (p := fd_option(s, k, model.r, model.q, model.sigma, t, False, True, g, g)["price"]),
                        "abs_error": abs(p - ref)} for g in grids])
    fd_order = float(np.polyfit(np.log(fd["grid"][:-1]), np.log(fd["abs_error"][:-1]), 1)[0])
    return {"tree": tree, "tree_order": tree_order, "tree_oscillation": max(odd) - min(odd), "mc": mc,
            "mc_order": mc_order, "fd": fd, "fd_order": fd_order, "american_reference": ref}


def greeks_comparison(model: FrontOfficePricer, mc_paths: int = 200_000, fd_grid: int = FD_GRID) -> pd.DataFrame:
    """Analytic European Greeks vs Monte Carlo common-random-number differences; American tree Greeks vs PDE."""
    rows = []
    for k in (90.0, 100.0, 110.0):
        for t in (0.5, 1.0):
            a = bs_greeks(SPOT, k, model.r, model.q, model.sigma, t, True)
            m = mc_greeks_crn(SPOT, k, model.r, model.q, model.sigma, t, True, mc_paths)
            for g in ("delta", "gamma", "vega"):
                rows.append({"model": "European call", "greek": g, "strike": k, "maturity": t, "front_office": a[g],
                             "challenger": m[g], "abs_diff": abs(a[g] - m[g]), "rel_diff": abs(a[g] - m[g]) / abs(a[g])})
            tr = crr_greeks(SPOT, k, model.r, model.q, model.sigma, t, model.tree_steps, False, True)
            fd = fd_option(SPOT, k, model.r, model.q, model.sigma, t, False, True, fd_grid, fd_grid)
            for g in ("delta", "gamma"):
                rows.append({"model": "American put", "greek": g, "strike": k, "maturity": t, "front_office": tr[g],
                             "challenger": fd[g], "abs_diff": abs(tr[g] - fd[g]),
                             "rel_diff": abs(tr[g] - fd[g]) / abs(fd[g])})
    return pd.DataFrame(rows)
