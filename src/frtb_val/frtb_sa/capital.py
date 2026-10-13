"""Standardised approach total: SBM + default risk charge + residual risk add-on (MAR20.4)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .. import columns as C
from .drc import DRCResult, drc_capital, rrao_capital
from .sbm import SCENARIOS, SBMResult, sbm_capital


@dataclass
class SAResult:
    sbm: SBMResult
    drc: DRCResult
    rrao: float
    total: float
    decomposition: pd.DataFrame   # risk class x measure for the binding scenario, plus DRC, RRAO and total


def sa_capital(sens: pd.DataFrame, curv: pd.DataFrame, drc_pos: pd.DataFrame, rrao_pos: pd.DataFrame) -> SAResult:
    sbm = sbm_capital(sens, curv)
    drc = drc_capital(drc_pos)
    rrao = rrao_capital(rrao_pos)
    rows = []
    for _, r in sbm.by_scenario.iterrows():
        rows.append({C.RISK_CLASS: r[C.RISK_CLASS], C.MEASURE: r[C.MEASURE], "charge": r[sbm.selected],
                     **{f"{s}_scenario": r[s] for s in SCENARIOS}})
    rows.append({C.RISK_CLASS: "DRC", C.MEASURE: C.DRC, "charge": drc.total})
    rows.append({C.RISK_CLASS: "RRAO", C.MEASURE: C.RRAO, "charge": rrao})
    dec = pd.DataFrame(rows)
    total = sbm.charge + drc.total + rrao
    dec.loc[len(dec)] = {C.RISK_CLASS: "TOTAL", C.MEASURE: "SA capital", "charge": total}
    return SAResult(sbm, drc, rrao, total, dec)
