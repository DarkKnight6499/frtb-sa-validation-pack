"""IMA expected shortfall: liquidity-horizon cascade, stressed ES and IMCC (MAR33.4 to MAR33.15)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..mathutil import expected_shortfall

LIQUIDITY_HORIZONS = (10, 20, 40, 60, 120)    # MAR33.4 Table 1
BASE_HORIZON = 10
ALPHA = 0.975                                  # MAR33.3
IMCC_RHO = 0.5                                 # MAR33.15(2)
WINDOW = 250                                   # twelve months of daily observations


def cascade_es(contrib: np.ndarray, lh: np.ndarray, alpha: float = ALPHA, base: int = BASE_HORIZON,
               horizons: tuple[int, ...] = LIQUIDITY_HORIZONS) -> float:
    """MAR33.4: ES = sqrt( ES_T(P)^2 + sum_{j>=2} ( ES_T(P, j) sqrt((LH_j - LH_{j-1}) / T) )^2 ).

    contrib holds base-horizon P&L contributions (scenarios x factors); lh the liquidity horizon per factor.
    """
    lh = np.asarray(lh)
    total = expected_shortfall(contrib.sum(axis=1), alpha) ** 2
    for j in range(1, len(horizons)):
        mask = lh >= horizons[j]
        if not mask.any():
            continue
        es_j = expected_shortfall(contrib[:, mask].sum(axis=1), alpha)
        total += (es_j * math.sqrt((horizons[j] - horizons[j - 1]) / base)) ** 2
    return math.sqrt(total)


def select_reduced_set(contrib_current: np.ndarray, lh: np.ndarray, broad: np.ndarray, fraction: float = 0.4,
                       floor: float = 0.75) -> list[int]:
    """Reduced factor set: the largest stand-alone-ES factors in each broad class, then top-ups to meet 75%.

    Taking a share of every class keeps hedges inside the set; ranking factors by stand-alone ES alone can pick
    one unhedged leg and overstate ES_R,C. MAR33.5(2)(b) requires ES_R,C to be at least 75% of ES_F,C.
    """
    lh = np.asarray(lh)
    standalone = np.array([expected_shortfall(contrib_current[:, i]) for i in range(contrib_current.shape[1])])
    chosen: set[int] = set()
    for cls in sorted(set(broad)):
        idx = np.where(broad == cls)[0]
        take = max(2, int(math.ceil(fraction * len(idx))))
        chosen.update(int(i) for i in idx[np.argsort(-standalone[idx])][:take])
    full = cascade_es(contrib_current, lh)
    rest = [int(i) for i in np.argsort(-standalone) if int(i) not in chosen]
    sel = sorted(chosen)
    while cascade_es(contrib_current[:, sel], lh[sel]) / full < floor and rest:
        sel = sorted(sel + [rest.pop(0)])
    return sel


@dataclass
class StressResult:
    start: int                  # index of the first scenario of the stressed window
    es_stress: float
    es_by_start: np.ndarray     # ES for each candidate start
    starts: np.ndarray


def find_stress_window(contrib_reduced: np.ndarray, lh_reduced: np.ndarray, window: int = WINDOW,
                       step: int = 5) -> StressResult:
    """MAR33.7: the 12-month window that maximises the reduced-set ES (equally weighted observations)."""
    n = contrib_reduced.shape[0]
    starts = np.arange(0, n - window + 1, step)
    es = np.array([cascade_es(contrib_reduced[s:s + window], lh_reduced) for s in starts])
    best = int(np.argmax(es))
    return StressResult(int(starts[best]), float(es[best]), es, starts)


@dataclass
class IMAResult:
    es_rs: float
    es_fc: float
    es_rc: float
    ratio_raw: float
    ratio_applied: float
    es_stressed: float
    explained: float            # ES_R,C / ES_F,C, the MAR33.5(2)(b) 75% condition
    stress_start: int
    imcc: float
    imcc_unconstrained: float
    imcc_constrained_sum: float
    by_class: dict


def stressed_es(contrib: np.ndarray, lh: np.ndarray, reduced: list[int], broad: np.ndarray, window: int = WINDOW,
                step: int = 5, rho: float = IMCC_RHO) -> IMAResult:
    """MAR33.6: ES = ES_R,S * max(ES_F,C / ES_R,C, 1), plus the IMCC blend of MAR33.15."""
    lh = np.asarray(lh)
    cur = slice(contrib.shape[0] - window, contrib.shape[0])
    es_fc = cascade_es(contrib[cur], lh)
    es_rc = cascade_es(contrib[cur][:, reduced], lh[reduced])
    stress = find_stress_window(contrib[:, reduced], lh[reduced], window, step)
    sl = slice(stress.start, stress.start + window)
    ratio_raw = es_fc / es_rc
    ratio = max(ratio_raw, 1.0)
    unconstrained = stress.es_stress * ratio
    by_class, constrained = {}, 0.0
    for cls in sorted(set(broad)):
        idx = np.where(broad == cls)[0]
        red_i = [i for i in reduced if broad[i] == cls]
        if not red_i:
            by_class[cls] = 0.0
            continue
        es_fc_i = cascade_es(contrib[cur][:, idx], lh[idx])
        es_rc_i = cascade_es(contrib[cur][:, red_i], lh[red_i])
        es_rs_i = cascade_es(contrib[sl][:, red_i], lh[red_i])
        by_class[cls] = es_rs_i * max(es_fc_i / es_rc_i, 1.0)
        constrained += by_class[cls]
    imcc = rho * unconstrained + (1 - rho) * constrained
    return IMAResult(stress.es_stress, es_fc, es_rc, ratio_raw, ratio, unconstrained, es_rc / es_fc, stress.start,
                     imcc, unconstrained, constrained, by_class)
