"""Default risk charge for non-securitisations (MAR22.9 to MAR22.26) and the residual risk add-on (MAR23)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .. import columns as C
from . import params as P

GROSS = "gross_jtd"
SCALED = "scaled_jtd"


def gross_jtd(pos: pd.DataFrame) -> pd.DataFrame:
    """MAR22.11 and MAR22.12: JTD = max(LGD x notional + P&L, 0) for longs, min(...) for shorts.

    MAR22.15 and MAR22.18 then scale exposures shorter than one year by maturity, floored at 3 months.
    """
    df = pos.copy()
    lgd = df[C.SENIORITY].map(P.DRC_LGD)
    raw = lgd * df[C.NOTIONAL] + df[C.PNL_TO_DATE]
    df[C.LGD] = lgd
    df[GROSS] = [max(x, 0.0) if n > 0 else min(x, 0.0) for x, n in zip(raw, df[C.NOTIONAL])]
    weight = df[C.MATURITY].clip(lower=P.DRC_MIN_MATURITY_WEIGHT, upper=1.0)
    df[SCALED] = df[GROSS] * weight
    return df


def net_jtd(gross: pd.DataFrame) -> pd.DataFrame:
    """MAR22.19: offset shorts against longs of the same obligor when the short is of equal or lower seniority."""
    rows = []
    for obligor, grp in gross.groupby(C.OBLIGOR):
        longs = [[P.DRC_SENIORITY_RANK[r[C.SENIORITY]], r[SCALED]] for _, r in grp.iterrows() if r[SCALED] > 0]
        shorts = [[P.DRC_SENIORITY_RANK[r[C.SENIORITY]], -r[SCALED]] for _, r in grp.iterrows() if r[SCALED] < 0]
        for s in sorted(shorts, key=lambda x: x[0]):
            for lg in sorted(longs, key=lambda x: -x[0]):       # offset the least senior eligible long first
                if lg[0] <= s[0] and s[1] > 0 and lg[1] > 0:
                    amt = min(lg[1], s[1])
                    lg[1] -= amt
                    s[1] -= amt
        first = grp.iloc[0]
        rows.append({C.OBLIGOR: obligor, C.DRC_BUCKET: first[C.DRC_BUCKET], C.RATING: first[C.RATING],
                     "net_long": sum(x[1] for x in longs), "net_short": sum(x[1] for x in shorts)})
    return pd.DataFrame(rows, columns=[C.OBLIGOR, C.DRC_BUCKET, C.RATING, "net_long", "net_short"])


@dataclass
class DRCResult:
    total: float
    by_bucket: pd.DataFrame
    net: pd.DataFrame


def drc_capital(pos: pd.DataFrame) -> DRCResult:
    """MAR22.22 to MAR22.26: hedge benefit ratio, risk-weighted long less HBR times short, summed over buckets."""
    empty = pd.DataFrame(columns=[C.DRC_BUCKET, "net_long", "net_short", "hbr", "rw_long", "rw_short", C.DRC])
    if pos.empty:
        return DRCResult(0.0, empty, pd.DataFrame())
    net = net_jtd(gross_jtd(pos))
    net["rw"] = net[C.RATING].str.upper().map(P.DRC_RISK_WEIGHTS)
    rows = []
    for bucket, grp in net.groupby(C.DRC_BUCKET):
        long_sum, short_sum = grp["net_long"].sum(), grp["net_short"].sum()
        denom = long_sum + short_sum
        hbr = long_sum / denom if denom > 0 else 0.0
        rw_long = float((grp["net_long"] * grp["rw"]).sum())
        rw_short = float((grp["net_short"] * grp["rw"]).sum())
        rows.append({C.DRC_BUCKET: bucket, "net_long": long_sum, "net_short": short_sum, "hbr": hbr,
                     "rw_long": rw_long, "rw_short": rw_short, C.DRC: max(rw_long - hbr * rw_short, 0.0)})
    by_bucket = pd.DataFrame(rows)
    return DRCResult(float(by_bucket[C.DRC].sum()), by_bucket, net)


def rrao_capital(pos: pd.DataFrame) -> float:
    """MAR23.8: gross notional times 1.0% (exotic underlying) or 0.1% (other residual risk)."""
    if pos.empty:
        return 0.0
    return float((pos[C.NOTIONAL] * pos["rrao_type"].map(P.RRAO_RW)).sum())
