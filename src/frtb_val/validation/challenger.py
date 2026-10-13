"""Challenger models: Monte Carlo with a control variate and a Crank-Nicolson finite-difference American pricer."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..mathutil import norm_ppf


@dataclass
class MCResult:
    price: float
    stderr: float
    plain_price: float
    plain_stderr: float
    n_paths: int

    @property
    def ci95(self) -> tuple[float, float]:
        return self.price - 1.96 * self.stderr, self.price + 1.96 * self.stderr

    @property
    def variance_reduction(self) -> float:
        return (self.plain_stderr / self.stderr) ** 2 if self.stderr > 0 else float("inf")


def mc_european(s: float, k: float, r: float, q: float, sigma: float, t: float, call: bool = True,
                n_paths: int = 100_000, seed: int = 1234, z: np.ndarray | None = None) -> MCResult:
    """European option by Monte Carlo with antithetic draws and the discounted terminal price as control variate.

    The control has known mean S exp(-q t); the coefficient is estimated by regression, so the estimator is
    consistent and its standard error comes from the regression residual.
    """
    if z is None:
        half = n_paths // 2
        base = np.random.default_rng(seed).standard_normal(half)
        z = np.concatenate([base, -base])
    st = s * np.exp((r - q - 0.5 * sigma ** 2) * t + sigma * math.sqrt(t) * z)
    disc = math.exp(-r * t)
    payoff = disc * (np.maximum(st - k, 0.0) if call else np.maximum(k - st, 0.0))
    control = disc * st
    mean_control = s * math.exp(-q * t)
    # Antithetic pairs are averaged first so the standard error treats each pair as one observation.
    m = len(z) // 2
    pay_pair = 0.5 * (payoff[:m] + payoff[m:2 * m])
    ctl_pair = 0.5 * (control[:m] + control[m:2 * m])
    cov = np.cov(pay_pair, ctl_pair, ddof=1)
    beta = cov[0, 1] / cov[1, 1]
    adjusted = pay_pair - beta * (ctl_pair - mean_control)
    return MCResult(float(adjusted.mean()), float(adjusted.std(ddof=1) / math.sqrt(m)), float(pay_pair.mean()),
                    float(pay_pair.std(ddof=1) / math.sqrt(m)), len(z))


def mc_greeks_crn(s: float, k: float, r: float, q: float, sigma: float, t: float, call: bool = True,
                  n_paths: int = 200_000, seed: int = 99) -> dict:
    """Delta, gamma and vega by central differences of the MC-CV price using common random numbers."""
    half = n_paths // 2
    base = np.random.default_rng(seed).standard_normal(half)
    z = np.concatenate([base, -base])

    def px(ss: float, vv: float) -> float:
        return mc_european(ss, k, r, q, vv, t, call, z=z).price

    hs, hv = 0.01 * s, 0.005
    up, mid, dn = px(s + hs, sigma), px(s, sigma), px(s - hs, sigma)
    return {"delta": (up - dn) / (2 * hs), "gamma": (up - 2 * mid + dn) / hs ** 2,
            "vega": (px(s, sigma + hv) - px(s, sigma - hv)) / (2 * hv)}


def _thomas_downward(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, payoff: np.ndarray,
                     x0: float, american: bool) -> np.ndarray:
    """Solve a tridiagonal system with optional projection (Brennan-Schwartz) for an early-exercise put.

    Nodes 1..m hold the unknowns, x0 is the known value below node 1 and the value above node m is already
    folded into d. Elimination runs from the top so back substitution starts in the exercise region.
    """
    m = len(b)
    bb, dd = b.copy(), d.copy()
    for i in range(m - 2, -1, -1):
        w = c[i] / bb[i + 1]
        bb[i] -= w * a[i + 1]
        dd[i] -= w * dd[i + 1]
    x = np.empty(m)
    prev = x0
    for i in range(m):
        val = (dd[i] - a[i] * prev) / bb[i]
        if american:
            val = max(val, payoff[i])
        x[i] = val
        prev = val
    return x


def fd_option(s: float, k: float, r: float, q: float, sigma: float, t: float, call: bool = False,
              american: bool = True, n_space: int = 800, n_time: int = 800) -> dict:
    """Crank-Nicolson PDE in log-price with Rannacher start-up; American puts use Brennan-Schwartz projection.

    American calls are not supported by the projection used here (exercise region is at high S), so they are
    rejected; with q = 0 an American call equals the European call.
    """
    if american and call:
        raise ValueError("American calls are not supported by the projection scheme")
    width = 6.0 * sigma * math.sqrt(t) + 0.3
    lo = min(math.log(s), math.log(k)) - width
    hi = max(math.log(s), math.log(k)) + width
    dx = (hi - lo) / n_space
    left = int(round((math.log(s) - lo) / dx))
    lo = math.log(s) - left * dx
    x = lo + dx * np.arange(n_space + 1)
    ss = np.exp(x)
    payoff = np.maximum(ss - k, 0.0) if call else np.maximum(k - ss, 0.0)
    v = payoff.copy()
    dt = t / n_time
    mu = r - q - 0.5 * sigma ** 2
    lower = 0.5 * sigma ** 2 / dx ** 2 - mu / (2 * dx)
    diag = -sigma ** 2 / dx ** 2 - r
    upper = 0.5 * sigma ** 2 / dx ** 2 + mu / (2 * dx)
    interior = slice(1, n_space)
    tau = 0.0

    def boundaries(tau_now: float) -> tuple[float, float]:
        if call:
            return 0.0, ss[-1] * math.exp(-q * tau_now) - k * math.exp(-r * tau_now)
        low = k - ss[0] if american else k * math.exp(-r * tau_now) - ss[0] * math.exp(-q * tau_now)
        return low, 0.0

    steps = [(1.0, dt / 2)] * 2 + [(0.5, dt)] * (n_time - 1)       # Rannacher: two half implicit steps first
    for theta, h in steps:
        tau += h
        lo_b, hi_b = boundaries(tau)
        rhs = v[interior] + (1 - theta) * h * (lower * v[0:-2] + diag * v[1:-1] + upper * v[2:])
        a = np.full(n_space - 1, -theta * h * lower)
        b = np.full(n_space - 1, 1 - theta * h * diag)
        c = np.full(n_space - 1, -theta * h * upper)
        rhs[-1] -= c[-1] * hi_b
        sol = _thomas_downward(a, b, c, rhs, payoff[interior], lo_b, american)
        v[interior] = sol
        v[0], v[-1] = lo_b, hi_b
    price = float(v[left])
    delta = float((v[left + 1] - v[left - 1]) / (ss[left + 1] - ss[left - 1]))
    h1, h2 = ss[left + 1] - ss[left], ss[left] - ss[left - 1]
    gamma = float(2 * ((v[left + 1] - v[left]) / h1 - (v[left] - v[left - 1]) / h2) / (h1 + h2))
    return {"price": price, "delta": delta, "gamma": gamma}


def mc_stderr_scaling(s: float, k: float, r: float, q: float, sigma: float, t: float,
                      paths: tuple[int, ...] = (2 ** 10, 2 ** 12, 2 ** 14, 2 ** 16, 2 ** 18)) -> list[dict]:
    """Standard error against path count, plain versus control variate."""
    out = []
    for n in paths:
        res = mc_european(s, k, r, q, sigma, t, True, n_paths=n, seed=7)
        out.append({"paths": n, "price": res.price, "stderr_cv": res.stderr, "stderr_plain": res.plain_stderr})
    return out


def z_quantile(p: float) -> float:
    return norm_ppf(p)
