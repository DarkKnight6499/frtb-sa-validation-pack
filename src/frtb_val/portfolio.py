"""Seeded synthetic trading book and bump-and-revalue sensitivities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import columns as C
from .frtb_sa import params as P
from .instruments import (CDS, CSR_BUCKET, EQ_BUCKET, IRS, Bond, EqOption, Exotic, FXForward, FXOption,
                          Instrument, Stock)
from .market import CSR_TENORS, GIRR_TENORS, VOL_TENORS, Market, build_market

GIRR_BUMP = 1e-4        # MAR21.19: 1 basis point
CSR_BUMP = 1e-4         # MAR21.20
REL_BUMP = 0.01         # MAR21.21, MAR21.24: 1 percentage point relative shift
VOL_REL_BUMP = 0.01     # relative vol bump, so s = (V(1.01 sigma) - V)/0.01 = vega * sigma (MAR21.25)
GAMMA_ABS_STEP = 1e-3   # 10bp central difference for second-order terms
GAMMA_REL_STEP = 0.01


def build_book(seed: int = 7) -> list[Instrument]:
    """Synthetic book of about 50 instruments. Sizes are seeded jitter around round numbers."""
    rng = np.random.default_rng(seed)

    def jit(x: float) -> float:
        return float(round(x * rng.uniform(0.85, 1.15), -3))

    book: list[Instrument] = []
    add = book.append
    counter = iter(range(1, 1000))

    def nid(prefix: str) -> str:
        return f"{prefix}{next(counter):03d}"

    for ccy, notional, fixed, mat, payer in [
            ("USD", 150e6, 0.0415, 10, True), ("USD", 80e6, 0.0400, 5, False), ("USD", 60e6, 0.0390, 3, True),
            ("USD", 40e6, 0.0420, 7, False), ("USD", 25e6, 0.0430, 30, True), ("EUR", 100e6, 0.0270, 5, True),
            ("EUR", 70e6, 0.0260, 10, False), ("GBP", 45e6, 0.0450, 2, True)]:
        add(IRS(nid("IRS"), "RATES", ccy=ccy, notional=jit(notional), fixed=fixed, maturity=mat, payer=payer))
    for face, cpn, mat in [(50e6, 0.040, 3), (35e6, 0.0425, 7), (-20e6, 0.0435, 10)]:
        add(Bond(nid("UST"), "RATES", face=jit(face), coupon=cpn, maturity=mat))
    for iss, face, cpn, mat in [("FIN_A", 20e6, 0.052, 5), ("IND_A", 15e6, 0.055, 7), ("TEC_A", 10e6, 0.050, 3),
                                ("FIN_HY", 8e6, 0.085, 5), ("CNS_HY", -5e6, 0.090, 3), ("SOV_A", 20e6, 0.045, 10)]:
        add(Bond(nid("BND"), "CREDIT", face=jit(face), coupon=cpn, maturity=mat, issuer=iss))
    for name, notional, cpn, mat, buyer, index in [
            ("IG_INDEX", 100e6, 0.01, 5, True, True), ("FIN_A", 20e6, 0.01, 5, True, False),
            ("IND_A", 10e6, 0.01, 5, False, False), ("FIN_HY", 5e6, 0.05, 5, True, False),
            ("CNS_A", 15e6, 0.01, 3, False, False), ("HC_A", 12e6, 0.01, 3, True, False)]:
        add(CDS(nid("CDS"), "CREDIT", name=name, notional=jit(notional), coupon=cpn, maturity=mat, buyer=buyer,
                index=index))
    for ccy, notional, strike, mat in [("EUR", 30e6, 1.09, 1.0), ("EUR", -20e6, 1.07, 0.5),
                                       ("GBP", 15e6, 1.28, 1.0), ("GBP", -10e6, 1.30, 2.0)]:
        add(FXForward(nid("FXF"), "FX", ccy=ccy, notional=jit(notional), strike=strike, maturity=mat))
    for ccy, notional, strike, mat, call in [("EUR", 8e6, 1.10, 1.0, True), ("EUR", -25e6, 1.05, 0.5, False),
                                             ("GBP", -20e6, 1.30, 2.0, True), ("GBP", 6e6, 1.25, 1.0, False)]:
        add(FXOption(nid("FXO"), "FX", ccy=ccy, notional=jit(notional), strike=strike, maturity=mat, call=call))
    for name, qty, strike, mat, call in [
            ("EQ_TECH", 20000, 190, 1.0, True), ("EQ_TECH", -15000, 170, 0.5, False), ("EQ_FIN", 30000, 95, 1.0, True),
            ("EQ_CONS", 25000, 62, 1.0, False), ("EQ_ENER", -20000, 125, 0.5, True),
            ("EQ_INDEX", 1000, 4400, 1.0, False), ("EQ_SMALL", 40000, 28, 1.0, True)]:
        add(EqOption(nid("EQO"), "EQD", name=name, qty=jit(qty), strike=strike, maturity=mat, call=call))
    # Priced as a vanilla proxy; flagged for the RRAO as a path-dependent (barrier) option (MAR23.5(1)).
    add(EqOption(nid("EQO"), "EQD", name="EQ_TECH", qty=jit(10000), strike=200, maturity=3.0, call=True,
                 rrao="other"))
    for name, shares in [("EQ_TECH", 40000), ("EQ_FIN", -30000), ("EQ_CONS", 50000), ("EQ_ENER", 25000),
                         ("EQ_SMALL", 100000)]:
        add(Stock(nid("STK"), "EQD", name=name, shares=jit(shares)))
    add(Exotic(nid("EXO"), "EQD", rrao="exotic", notional=40e6))   # longevity-type underlying (MAR23.3 footnote 1)
    add(Exotic(nid("EXO"), "EQD", rrao="exotic", notional=15e6))   # weather-type underlying
    return book


def _gamma(inst: Instrument, m: Market, kind: str, key: str, idx: int | None, step: float, rel: bool,
           base: float) -> float:
    up = inst.value(m.bumped(kind, key, idx, step, relative=rel))
    dn = inst.value(m.bumped(kind, key, idx, -step, relative=rel))
    return (up - 2 * base + dn) / step ** 2


def _factor_rows(inst: Instrument, m: Market, kind: str, key: str, base: float) -> list[dict]:
    """Delta, vega and diagonal-gamma rows for one risk-factor group of one instrument."""
    rows: list[dict] = []
    ident = {C.INSTRUMENT: inst.iid, C.DESK: inst.desk}
    if kind == "girr":
        for i, t in enumerate(GIRR_TENORS):
            s = (inst.value(m.bumped("girr", key, i, GIRR_BUMP)) - base) / GIRR_BUMP
            g = _gamma(inst, m, "girr", key, i, GAMMA_ABS_STEP, False, base)
            rows.append({**ident, C.RISK_CLASS: C.GIRR, C.MEASURE: C.DELTA, C.BUCKET: key, C.RISK_FACTOR: key,
                         C.CURVE: "YIELD", C.TENOR: float(t), C.SENSITIVITY: s, C.GAMMA: g})
    elif kind == "csr":
        for i, t in enumerate(CSR_TENORS):
            s = (inst.value(m.bumped("csr", key, i, CSR_BUMP)) - base) / CSR_BUMP
            g = _gamma(inst, m, "csr", key, i, GAMMA_ABS_STEP, False, base)
            rows.append({**ident, C.RISK_CLASS: C.CSR, C.MEASURE: C.DELTA, C.BUCKET: CSR_BUCKET[key],
                         C.RISK_FACTOR: key, C.CURVE: "CDS" if isinstance(inst, CDS) else "BOND",
                         C.TENOR: float(t), C.SENSITIVITY: s, C.GAMMA: g})
    elif kind in ("eq", "fx"):
        s = (inst.value(m.bumped(kind, key, None, REL_BUMP, relative=True)) - base) / REL_BUMP
        g = _gamma(inst, m, kind, key, None, GAMMA_REL_STEP, True, base)
        cls, bucket = (C.EQ, EQ_BUCKET[key]) if kind == "eq" else (C.FX, key)
        rows.append({**ident, C.RISK_CLASS: cls, C.MEASURE: C.DELTA, C.BUCKET: bucket, C.RISK_FACTOR: key,
                     C.CURVE: "SPOT", C.TENOR: np.nan, C.SENSITIVITY: s, C.GAMMA: g})
    elif kind in ("vol_eq", "vol_fx"):
        for i, t in enumerate(VOL_TENORS):
            s = (inst.value(m.bumped(kind, key, i, VOL_REL_BUMP, relative=True)) - base) / VOL_REL_BUMP
            g = _gamma(inst, m, kind, key, i, GAMMA_REL_STEP, True, base)
            cls, bucket = (C.EQ, EQ_BUCKET[key]) if kind == "vol_eq" else (C.FX, key)
            rows.append({**ident, C.RISK_CLASS: cls, C.MEASURE: C.VEGA, C.BUCKET: bucket, C.RISK_FACTOR: key,
                         C.CURVE: "VOL", C.TENOR: float(t), C.SENSITIVITY: s, C.GAMMA: g})
    return rows


def sensitivities(book: list[Instrument], m: Market) -> pd.DataFrame:
    """Instrument-level delta and vega sensitivities (MAR21.15 to MAR21.26) with diagonal gamma."""
    rows: list[dict] = []
    for inst in book:
        base = inst.value(m)
        for kind, key in inst.exposures():
            rows.extend(_factor_rows(inst, m, kind, key, base))
    cols = [C.INSTRUMENT, C.DESK, C.RISK_CLASS, C.MEASURE, C.BUCKET, C.RISK_FACTOR, C.CURVE, C.TENOR,
            C.SENSITIVITY, C.GAMMA]
    df = pd.DataFrame(rows, columns=cols)
    keep = (df[C.SENSITIVITY].abs() > 0) | (df[C.GAMMA].abs() > 0)
    return df[keep].reset_index(drop=True)


def curvature(book: list[Instrument], m: Market, include_linear: bool = False) -> pd.DataFrame:
    """Curvature risk positions CVR+ and CVR- per risk factor (MAR21.5, MAR21.98, MAR21.99).

    Only instruments with optionality are shocked unless include_linear is set (MAR21.2(3), (4)).
    """
    rows: list[dict] = []
    for inst in book:
        if not (inst.is_option or include_linear):
            continue
        base = inst.value(m)
        for kind, key in inst.exposures():
            if kind in ("vol_eq", "vol_fx"):
                continue
            if kind in ("girr", "csr"):
                if kind == "girr":
                    rw, cls, bucket, bump = P.GIRR_CURVATURE_RW, C.GIRR, key, GIRR_BUMP
                else:
                    bucket = CSR_BUCKET[key]
                    rw, cls, bump = P.CSR_RISK_WEIGHTS[bucket], C.CSR, CSR_BUMP
                up = inst.value(m.bumped_parallel(kind, key, rw))
                dn = inst.value(m.bumped_parallel(kind, key, -rw))
                s = (inst.value(m.bumped_parallel(kind, key, bump)) - base) / bump   # sum of tenor deltas
            else:
                if kind == "eq":
                    bucket = EQ_BUCKET[key]
                    rw, cls = P.EQ_RISK_WEIGHTS[bucket], C.EQ
                else:
                    bucket, rw, cls = key, P.FX_RW, C.FX
                up = inst.value(m.bumped(kind, key, None, rw, relative=True))
                dn = inst.value(m.bumped(kind, key, None, -rw, relative=True))
                s = (inst.value(m.bumped(kind, key, None, REL_BUMP, relative=True)) - base) / REL_BUMP
            rows.append({C.RISK_CLASS: cls, C.BUCKET: bucket, C.RISK_FACTOR: key,
                         C.CVR_UP: -(up - base - rw * s), C.CVR_DOWN: -(dn - base + rw * s)})
    cols = [C.RISK_CLASS, C.BUCKET, C.RISK_FACTOR, C.CVR_UP, C.CVR_DOWN]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    keys = [C.RISK_CLASS, C.BUCKET, C.RISK_FACTOR]
    return df.groupby(keys, as_index=False)[[C.CVR_UP, C.CVR_DOWN]].sum()


def drc_positions(book: list[Instrument], m: Market) -> pd.DataFrame:
    items = [it for inst in book for it in inst.drc_items(m)]
    cols = [C.OBLIGOR, C.SENIORITY, C.RATING, C.DRC_BUCKET, C.NOTIONAL, C.PNL_TO_DATE, C.MATURITY, "kind"]
    return pd.DataFrame(items, columns=cols)


def rrao_positions(book: list[Instrument], m: Market) -> pd.DataFrame:
    rows = [{C.INSTRUMENT: i.iid, "rrao_type": i.rrao, C.NOTIONAL: i.rrao_notional(m)} for i in book if i.rrao]
    return pd.DataFrame(rows, columns=[C.INSTRUMENT, "rrao_type", C.NOTIONAL])


def book_value(book: list[Instrument], m: Market) -> float:
    return float(sum(i.value(m) for i in book))


def default_book_and_market(seed: int = 7) -> tuple[list[Instrument], Market]:
    return build_book(seed), build_market()
