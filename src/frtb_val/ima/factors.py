"""Risk-factor table, liquidity horizon mapping and a seeded synthetic factor-change panel."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .. import columns as C

FACTOR_ID = "factor_id"
LIQ_HORIZON = "liquidity_horizon"
BROAD_CLASS = "broad_class"
SPECIFIED = {"USD", "EUR", "GBP"}        # MAR33.12 Table 2: specified currencies / currency pairs vs USD

_BROAD = {C.GIRR: "IR", C.EQ: "EQ", C.FX: "FX", C.CSR: "CS"}


def liquidity_horizon(risk_class: str, measure: str, bucket) -> int:
    """MAR33.12 Table 2 mapping for the synthetic book (names, buckets and currencies are stylised)."""
    if risk_class == C.GIRR:
        if measure == C.VEGA:
            return 60
        return 10 if bucket in SPECIFIED else 20
    if risk_class == C.CSR:
        if measure == C.VEGA:
            return 120
        b = int(bucket)
        if b == 1:
            return 20          # sovereign IG
        if b == 9:
            return 40          # sovereign HY
        return 60 if 10 <= b <= 16 or b == 18 else 40   # corporate HY 60, corporate IG and IG index 40
    if risk_class == C.EQ:
        b = int(bucket)
        if measure == C.VEGA:
            return 20 if b in (1, 2, 3, 4, 5, 6, 7, 8, 12, 13) else 60
        if b in (1, 2, 3, 4, 5, 6, 7, 8, 12, 13):
            return 10
        return 20 if b in (9, 10) else 60
    if risk_class == C.FX:
        return 40 if measure == C.VEGA else 10
    raise ValueError(risk_class)


def factor_table(sens: pd.DataFrame, desk: str | None = None) -> pd.DataFrame:
    """Net delta, vega and gamma by risk factor, optionally for one desk."""
    df = sens if desk is None else sens[sens[C.DESK] == desk]
    df = df.copy()
    df[C.TENOR] = df[C.TENOR].fillna(-1.0)
    keys = [C.RISK_CLASS, C.MEASURE, C.BUCKET, C.RISK_FACTOR, C.CURVE, C.TENOR]
    out = df.groupby(keys, as_index=False)[[C.SENSITIVITY, C.GAMMA]].sum()
    out[C.TENOR] = out[C.TENOR].where(out[C.TENOR] >= 0, np.nan)
    out[FACTOR_ID] = [f"{r[C.RISK_CLASS]}|{r[C.MEASURE]}|{r[C.BUCKET]}|{r[C.RISK_FACTOR]}|{r[C.CURVE]}|"
                      f"{'' if pd.isna(r[C.TENOR]) else r[C.TENOR]}" for _, r in out.iterrows()]
    out[LIQ_HORIZON] = [liquidity_horizon(r[C.RISK_CLASS], r[C.MEASURE], r[C.BUCKET]) for _, r in out.iterrows()]
    out[BROAD_CLASS] = out[C.RISK_CLASS].map(_BROAD)
    return out.reset_index(drop=True)


def daily_vol(row: pd.Series) -> float:
    """Assumed daily factor-change volatility (absolute for rates and spreads, relative otherwise)."""
    rc, ms, b = row[C.RISK_CLASS], row[C.MEASURE], row[C.BUCKET]
    if rc == C.GIRR:
        return 0.0007
    if rc == C.CSR:
        return 0.0010 if int(b) in (11, 12, 13, 14, 15, 16, 18) else 0.00025
    if rc == C.EQ:
        if ms == C.VEGA:
            return 0.03
        return {12: 0.009, 10: 0.020}.get(int(b), 0.013)
    if rc == C.FX:
        return 0.025 if ms == C.VEGA else 0.0055
    raise ValueError(rc)


def _pair_corr(a: pd.Series, b: pd.Series) -> float:
    ra, rb = a[C.RISK_CLASS], b[C.RISK_CLASS]
    if ra == rb:
        if ra == C.GIRR:
            same_ccy = a[C.BUCKET] == b[C.BUCKET]
            ta, tb = a[C.TENOR], b[C.TENOR]
            base = 1.0 if (np.isnan(ta) or np.isnan(tb)) else max(math.exp(-0.03 * abs(ta - tb) / min(ta, tb)), 0.4)
            return base if same_ccy else 0.4 * base
        if a[C.MEASURE] != b[C.MEASURE]:
            return -0.6 if (ra == C.EQ and a[C.RISK_FACTOR] == b[C.RISK_FACTOR]) else -0.3
        if ra == C.CSR:
            return 0.7 if a[C.RISK_FACTOR] == b[C.RISK_FACTOR] else (0.5 if a[C.BUCKET] == b[C.BUCKET] else 0.35)
        if ra == C.EQ:
            return 0.7 if a[C.MEASURE] == C.VEGA else 0.55
        return 0.6 if a[C.MEASURE] == C.VEGA else 0.3
    pair = {ra, rb}
    return {frozenset({C.GIRR, C.CSR}): -0.1, frozenset({C.GIRR, C.EQ}): 0.1, frozenset({C.CSR, C.EQ}): -0.4,
            frozenset({C.EQ, C.FX}): 0.1}.get(frozenset(pair), 0.0)


def correlation_matrix(meta: pd.DataFrame) -> np.ndarray:
    n = len(meta)
    rows = [meta.iloc[i] for i in range(n)]
    r = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r[i, j] = r[j, i] = _pair_corr(rows[i], rows[j])
    return _nearest_psd(r)


def _nearest_psd(r: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    w, v = np.linalg.eigh(r)
    out = (v * np.maximum(w, floor)) @ v.T
    d = np.sqrt(np.diag(out))
    return out / np.outer(d, d)


def simulate_panel(meta: pd.DataFrame, n_days: int = 2500, stress_start: int = 700, stress_len: int = 250,
                   stress_vol_mult: float = 2.5, stress_corr_push: float = 0.4, df_t: int = 5,
                   seed: int = 11) -> pd.DataFrame:
    """Synthetic daily factor changes: correlated Student-t, with a planted high-vol, high-correlation window.

    The planted window is the only stress regime, so the stress-period search has a known answer to recover.
    """
    rng = np.random.default_rng(seed)
    k = len(meta)
    corr = correlation_matrix(meta)
    sign = np.sign(corr)
    corr_s = _nearest_psd(corr + stress_corr_push * (sign - corr) * (1 - np.eye(k)))
    vol = meta.apply(daily_vol, axis=1).to_numpy()
    chol, chol_s = np.linalg.cholesky(corr), np.linalg.cholesky(corr_s)
    z = rng.standard_normal((n_days, k))
    w = rng.chisquare(df_t, n_days) / df_t
    tscale = math.sqrt((df_t - 2) / df_t) / np.sqrt(w)
    x = np.empty((n_days, k))
    in_stress = np.zeros(n_days, dtype=bool)
    in_stress[stress_start:stress_start + stress_len] = True
    x[~in_stress] = z[~in_stress] @ chol.T * vol
    x[in_stress] = z[in_stress] @ chol_s.T * vol * stress_vol_mult
    x *= tscale[:, None]
    return pd.DataFrame(x, columns=meta[FACTOR_ID].tolist())


def horizon_changes(x: np.ndarray, h: int) -> np.ndarray:
    """Overlapping h-day sums of daily changes (MAR33.4(7))."""
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    return c[h:] - c[:-h]


def contributions(x: np.ndarray, s: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Per-factor P&L contribution under a delta-gamma approximation: s dx + 0.5 g dx^2."""
    return s * x + 0.5 * g * x ** 2
