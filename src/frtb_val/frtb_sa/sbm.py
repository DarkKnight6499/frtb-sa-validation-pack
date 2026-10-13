"""Sensitivities-based method: delta, vega and curvature aggregation (MAR21.4 to MAR21.7)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import columns as C
from . import params as P

SCENARIOS = ("low", "medium", "high")
UND_TENOR = "underlying_tenor"   # optional column: underlying maturity of a GIRR vega factor (MAR21.93(2))


def scale_corr(rho: float | np.ndarray, scenario: str):
    """MAR21.6: scale a correlation parameter for the medium, high or low scenario."""
    if scenario == "medium":
        return rho
    if scenario == "high":
        return np.minimum(P.HIGH_MULT * rho, 1.0)
    if scenario == "low":
        return np.maximum(2.0 * rho - 1.0, P.LOW_FLOOR_MULT * rho)
    raise ValueError(scenario)


def _scale_matrix(r: np.ndarray, scenario: str) -> np.ndarray:
    out = scale_corr(r, scenario)
    np.fill_diagonal(out, 1.0)
    return out


# == risk weights

def risk_weight(risk_class: str, measure: str, bucket, tenor: float, curve: str = "YIELD") -> float:
    """Risk weight for one risk factor (MAR21.42, MAR21.43, MAR21.53, MAR21.77, MAR21.87, MAR21.92)."""
    if measure == C.DELTA:
        if risk_class == C.GIRR:
            if curve in ("INFLATION",) or str(curve).startswith("XCCY"):
                return P.GIRR_INFLATION_XCCY_RW
            return dict(zip(P.GIRR_TENORS, P.GIRR_RISK_WEIGHTS))[float(tenor)]
        if risk_class == C.CSR:
            return P.CSR_RISK_WEIGHTS[int(bucket)]
        if risk_class == C.EQ:
            rw = P.EQ_RISK_WEIGHTS[int(bucket)]
            return rw / 100.0 if curve == "REPO" else rw
        if risk_class == C.FX:
            return P.FX_RW
    if measure == C.VEGA:
        if risk_class == C.GIRR:
            return P.vega_risk_weight(P.VEGA_LIQUIDITY_HORIZON["GIRR"])
        if risk_class == C.CSR:
            return P.vega_risk_weight(P.VEGA_LIQUIDITY_HORIZON["CSR"])
        if risk_class == C.EQ:
            key = "EQ_LARGE" if int(bucket) in P.EQ_LARGE_CAP_VEGA_BUCKETS else "EQ_SMALL"
            return P.vega_risk_weight(P.VEGA_LIQUIDITY_HORIZON[key])
        if risk_class == C.FX:
            return P.vega_risk_weight(P.VEGA_LIQUIDITY_HORIZON["FX"])
    raise ValueError((risk_class, measure))


# == correlations

def _girr_delta_rho(a: pd.Series, b: pd.Series) -> float:
    ca, cb = str(a[C.CURVE]), str(b[C.CURVE])
    infl_a, infl_b = ca == "INFLATION", cb == "INFLATION"
    xc_a, xc_b = ca.startswith("XCCY"), cb.startswith("XCCY")
    if xc_a or xc_b:
        return P.GIRR_RHO_XCCY
    if infl_a != infl_b:
        return P.GIRR_RHO_INFLATION
    if infl_a and infl_b:
        return 1.0
    tk, tl = float(a[C.TENOR]), float(b[C.TENOR])
    same_curve = ca == cb
    if tk == tl:
        return 1.0 if same_curve else P.GIRR_RHO_SAME_TENOR_DIFF_CURVE
    rho = P.girr_tenor_rho(tk, tl)
    return rho if same_curve else rho * P.GIRR_RHO_SAME_TENOR_DIFF_CURVE


def _csr_name_rho(bucket: int) -> float:
    return P.CSR_RHO_NAME_INDEX if bucket in (17, 18) else P.CSR_RHO_NAME


def _eq_rho(bucket: int) -> float:
    return P.EQ_RHO[bucket]


def intra_bucket_rho(risk_class: str, measure: str, bucket, f: pd.DataFrame) -> np.ndarray:
    """Medium-scenario within-bucket correlation matrix (MAR21.45 to MAR21.49, 21.54, 21.55, 21.78, 21.93, 21.94)."""
    n = len(f)
    r = np.eye(n)
    rows = [f.iloc[i] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = rows[i], rows[j]
            if risk_class == C.GIRR and measure == C.DELTA:
                v = _girr_delta_rho(a, b)
            elif risk_class == C.GIRR and measure == C.VEGA:
                ropt = P.vega_maturity_rho(float(a[C.TENOR]), float(b[C.TENOR]))
                ua = float(a.get(UND_TENOR, a[C.TENOR])) if UND_TENOR in f.columns else float(a[C.TENOR])
                ub = float(b.get(UND_TENOR, b[C.TENOR])) if UND_TENOR in f.columns else float(b[C.TENOR])
                v = min(ropt * P.vega_maturity_rho(ua, ub), 1.0)
            elif risk_class == C.CSR:
                name = 1.0 if a[C.RISK_FACTOR] == b[C.RISK_FACTOR] else _csr_name_rho(int(bucket))
                if measure == C.DELTA:
                    tenor = 1.0 if float(a[C.TENOR]) == float(b[C.TENOR]) else P.CSR_RHO_TENOR
                    basis = 1.0 if a[C.CURVE] == b[C.CURVE] else P.CSR_RHO_BASIS
                    v = name * tenor * basis
                else:
                    v = min(name * P.vega_maturity_rho(float(a[C.TENOR]), float(b[C.TENOR])), 1.0)
            elif risk_class == C.EQ:
                same_name = a[C.RISK_FACTOR] == b[C.RISK_FACTOR]
                if measure == C.DELTA:
                    if same_name:
                        v = P.EQ_RHO_SPOT_REPO_SAME_NAME if a[C.CURVE] != b[C.CURVE] else 1.0
                    else:
                        v = _eq_rho(int(bucket)) * (P.EQ_RHO_SPOT_REPO_SAME_NAME if a[C.CURVE] != b[C.CURVE] else 1.0)
                else:
                    base = 1.0 if same_name else _eq_rho(int(bucket))
                    v = min(base * P.vega_maturity_rho(float(a[C.TENOR]), float(b[C.TENOR])), 1.0)
            elif risk_class == C.FX:
                v = P.vega_maturity_rho(float(a[C.TENOR]), float(b[C.TENOR])) if measure == C.VEGA else 1.0
            else:
                raise ValueError(risk_class)
            r[i, j] = r[j, i] = v
    return r


def inter_bucket_gamma(risk_class: str, b, c) -> float:
    if risk_class == C.GIRR:
        return 1.0 if b == c else P.GIRR_GAMMA
    if risk_class == C.FX:
        return 1.0 if b == c else P.FX_GAMMA
    if risk_class == C.CSR:
        return P.csr_gamma(int(b), int(c))
    if risk_class == C.EQ:
        return P.eq_gamma(int(b), int(c))
    raise ValueError(risk_class)


def _is_other_bucket(risk_class: str, bucket) -> bool:
    return (risk_class == C.CSR and bucket == P.CSR_OTHER_BUCKET) or \
           (risk_class == C.EQ and bucket == P.EQ_OTHER_BUCKET)


# == aggregation primitives

def bucket_capital(ws: np.ndarray, rho: np.ndarray) -> float:
    """MAR21.4(4): K_b = sqrt(max(0, sum WS^2 + sum_{k != l} rho WS_k WS_l))."""
    return math.sqrt(max(0.0, float(ws @ rho @ ws)))


def aggregate_buckets(k: dict, s: dict, gamma) -> float:
    """MAR21.4(5): across-bucket aggregation including the alternative specification when negative."""
    keys = list(k)
    total = sum(k[b] ** 2 for b in keys)
    cross = 0.0
    for i, b in enumerate(keys):
        for c in keys[i + 1:]:
            cross += 2.0 * gamma(b, c) * s[b] * s[c]
    if total + cross >= 0:
        return math.sqrt(total + cross)
    s_alt = {b: max(min(s[b], k[b]), -k[b]) for b in keys}
    cross = sum(2.0 * gamma(b, c) * s_alt[b] * s_alt[c] for i, b in enumerate(keys) for c in keys[i + 1:])
    return math.sqrt(max(0.0, total + cross))


def _psi(a: float, b: float) -> float:
    return 0.0 if (a < 0 and b < 0) else 1.0


def curvature_bucket(cvr_up: np.ndarray, cvr_dn: np.ndarray, rho: np.ndarray) -> tuple[float, float]:
    """MAR21.5(3): bucket curvature capital K_b and the selected-scenario sum S_b."""
    def k_of(cvr: np.ndarray) -> float:
        pos = np.maximum(cvr, 0.0)
        total = float(pos @ pos)
        n = len(cvr)
        for i in range(n):
            for j in range(n):
                if i != j:
                    total += rho[i, j] * cvr[i] * cvr[j] * _psi(cvr[i], cvr[j])
        return math.sqrt(max(0.0, total))

    k_up, k_dn = k_of(cvr_up), k_of(cvr_dn)
    if k_up > k_dn or (k_up == k_dn and cvr_up.sum() > cvr_dn.sum()):
        return k_up, float(cvr_up.sum())
    return k_dn, float(cvr_dn.sum())


def aggregate_curvature(k: dict, s: dict, gamma) -> float:
    """MAR21.5(4): across-bucket curvature aggregation with the psi sign rule."""
    keys = list(k)
    total = sum(k[b] ** 2 for b in keys)
    for i, b in enumerate(keys):
        for c in keys[i + 1:]:
            total += 2.0 * gamma(b, c) * s[b] * s[c] * _psi(s[b], s[c])
    return math.sqrt(max(0.0, total))


# == risk-class level capital

def net_sensitivities(sens: pd.DataFrame) -> pd.DataFrame:
    keys = [C.RISK_CLASS, C.MEASURE, C.BUCKET, C.RISK_FACTOR, C.CURVE, C.TENOR]
    extra = [UND_TENOR] if UND_TENOR in sens.columns else []
    df = sens.copy()
    df[C.TENOR] = df[C.TENOR].fillna(-1.0)   # spot factors have no tenor; keep them groupable
    out = df.groupby(keys + extra, as_index=False)[C.SENSITIVITY].sum()
    out[C.TENOR] = out[C.TENOR].where(out[C.TENOR] >= 0, np.nan)
    return out


def _weighted(net: pd.DataFrame) -> pd.DataFrame:
    out = net.copy()
    out["rw"] = [risk_weight(r[C.RISK_CLASS], r[C.MEASURE], r[C.BUCKET], r[C.TENOR], r[C.CURVE])
                 for _, r in out.iterrows()]
    out["ws"] = out["rw"] * out[C.SENSITIVITY]
    return out


def delta_vega_capital(net: pd.DataFrame, risk_class: str, measure: str, scenario: str) -> tuple[float, pd.DataFrame]:
    """Capital for one risk class and measure under a correlation scenario; also per-bucket K and S."""
    sub = net[(net[C.RISK_CLASS] == risk_class) & (net[C.MEASURE] == measure)]
    if sub.empty:
        return 0.0, pd.DataFrame(columns=[C.BUCKET, "K", "S"])
    w = _weighted(sub)
    k, s = {}, {}
    for bucket, grp in w.groupby(C.BUCKET, sort=True):
        ws = grp["ws"].to_numpy()
        if _is_other_bucket(risk_class, bucket):
            k[bucket] = float(np.abs(ws).sum())      # MAR21.56(1), MAR21.79(1)
        else:
            rho = _scale_matrix(intra_bucket_rho(risk_class, measure, bucket, grp.reset_index(drop=True)), scenario)
            k[bucket] = bucket_capital(ws, rho)
        s[bucket] = float(ws.sum())

    def gamma(b, c):
        g = inter_bucket_gamma(risk_class, b, c)
        return float(scale_corr(g, scenario))

    cap = aggregate_buckets(k, s, gamma)
    detail = pd.DataFrame({C.BUCKET: list(k), "K": list(k.values()), "S": [s[b] for b in k]})
    return cap, detail


def _curv_rho(risk_class: str, bucket, grp: pd.DataFrame) -> np.ndarray:
    """Within-bucket curvature correlation: squared delta correlation (MAR21.100)."""
    n = len(grp)
    r = np.eye(n)
    names = grp[C.RISK_FACTOR].tolist()
    for i in range(n):
        for j in range(i + 1, n):
            if risk_class == C.CSR:
                base = _csr_name_rho(int(bucket))              # MAR21.100(1): only the name correlation applies
            elif risk_class == C.EQ:
                base = _eq_rho(int(bucket)) if names[i] != names[j] else 1.0
            elif risk_class == C.GIRR:
                base = P.GIRR_RHO_SAME_TENOR_DIFF_CURVE
            else:
                base = 1.0
            r[i, j] = r[j, i] = base ** 2
    return r


def curvature_capital(curv: pd.DataFrame, risk_class: str, scenario: str) -> tuple[float, pd.DataFrame]:
    sub = curv[curv[C.RISK_CLASS] == risk_class]
    if sub.empty:
        return 0.0, pd.DataFrame(columns=[C.BUCKET, "K", "S"])
    k, s = {}, {}
    for bucket, grp in sub.groupby(C.BUCKET, sort=True):
        up, dn = grp[C.CVR_UP].to_numpy(), grp[C.CVR_DOWN].to_numpy()
        if _is_other_bucket(risk_class, bucket):
            # MAR21.56(2), MAR21.79(2)
            k_up, k_dn = float(np.maximum(up, 0).sum()), float(np.maximum(dn, 0).sum())
            k[bucket], s[bucket] = (k_up, float(up.sum())) if k_up >= k_dn else (k_dn, float(dn.sum()))
        else:
            rho = _scale_matrix(_curv_rho(risk_class, bucket, grp.reset_index(drop=True)), scenario)
            k[bucket], s[bucket] = curvature_bucket(up, dn, rho)

    def gamma(b, c):
        return float(scale_corr(inter_bucket_gamma(risk_class, b, c) ** 2, scenario))   # MAR21.101

    cap = aggregate_curvature(k, s, gamma)
    detail = pd.DataFrame({C.BUCKET: list(k), "K": list(k.values()), "S": [s[b] for b in k]})
    return cap, detail


@dataclass
class SBMResult:
    by_scenario: pd.DataFrame      # rows: risk class x measure, columns: scenario charges
    totals: dict                   # scenario -> total SBM charge
    selected: str                  # scenario with the largest total (MAR21.7)
    charge: float                  # max over scenarios


def sbm_capital(sens: pd.DataFrame, curv: pd.DataFrame | None = None) -> SBMResult:
    """Total SBM charge: sum delta, vega and curvature over risk classes per scenario, then take the max."""
    net = net_sensitivities(sens) if not sens.empty else sens
    curv = curv if curv is not None else pd.DataFrame(columns=[C.RISK_CLASS, C.BUCKET, C.RISK_FACTOR, C.CVR_UP,
                                                               C.CVR_DOWN])
    classes = [C.GIRR, C.CSR, C.EQ, C.FX]
    rows = []
    for rc in classes:
        for measure in (C.DELTA, C.VEGA, C.CURVATURE):
            row = {C.RISK_CLASS: rc, C.MEASURE: measure}
            for sc in SCENARIOS:
                if measure == C.CURVATURE:
                    row[sc] = curvature_capital(curv, rc, sc)[0]
                else:
                    row[sc] = delta_vega_capital(net, rc, measure, sc)[0] if not sens.empty else 0.0
            rows.append(row)
    table = pd.DataFrame(rows)
    totals = {sc: float(table[sc].sum()) for sc in SCENARIOS}
    selected = max(totals, key=totals.get)
    return SBMResult(table, totals, selected, totals[selected])
