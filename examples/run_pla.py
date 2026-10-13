"""P&L attribution test and desk-level eligibility on four synthetic desks."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# == config
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "examples" / "output"
BOOK_SEED = 7
PANEL_SEED = 11
NOISE_SEED = 2026
# Configured risk-model deficiencies per desk. These are illustrative assumptions, not measurements:
# which factor group the risk-theoretical model omits, whether it drops second-order terms, and the size of
# unexplained P&L (valuation or data differences) in each series as a share of the desk P&L volatility.
DESKS = {
    "RATES": {"omit_measure": None, "omit_gamma": False, "omit_tenor_min": None, "hpl_noise": 0.10, "rtpl_noise": 0.10},
    "FX": {"omit_measure": "vega", "omit_gamma": False, "omit_tenor_min": None, "hpl_noise": 0.10, "rtpl_noise": 0.10},
    "EQD": {"omit_measure": None, "omit_gamma": True, "omit_tenor_min": None, "hpl_noise": 0.30, "rtpl_noise": 0.30},
    "CREDIT": {"omit_measure": None, "omit_gamma": False, "omit_tenor_min": 5.0, "hpl_noise": 0.50, "rtpl_noise": 0.50},
}
# ==
sys.path.insert(0, str(ROOT / "src"))

from frtb_val import columns as C  # noqa: E402
from frtb_val.ima import backtest as B  # noqa: E402
from frtb_val.ima import factors as F  # noqa: E402
from frtb_val.ima.pipeline import build_inputs  # noqa: E402
from frtb_val.pla.pla_test import desk_eligibility, pla_test  # noqa: E402
from frtb_val.validation.report import md_table  # noqa: E402


def desk_series(inp, desk: str, cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Hypothetical and risk-theoretical daily P&L for one desk."""
    table = F.factor_table(inp.sens, desk)
    meta = inp.meta.set_index(F.FACTOR_ID)
    s = np.zeros(len(meta))
    g = np.zeros(len(meta))
    pos = {fid: i for i, fid in enumerate(meta.index)}
    for _, row in table.iterrows():
        s[pos[row[F.FACTOR_ID]]] = row[C.SENSITIVITY]
        g[pos[row[F.FACTOR_ID]]] = row[C.GAMMA]
    x = inp.panel.to_numpy()
    core = F.contributions(x, s, g).sum(axis=1)
    keep = np.ones(len(meta), dtype=bool)
    if cfg["omit_measure"]:
        keep &= (meta[C.MEASURE] != cfg["omit_measure"]).to_numpy()
    if cfg["omit_tenor_min"] is not None:
        tenor = meta[C.TENOR].to_numpy()
        keep &= ~(meta[C.RISK_CLASS].eq(C.CSR).to_numpy() & (np.nan_to_num(tenor, nan=0.0) >= cfg["omit_tenor_min"]))
    rt = F.contributions(x[:, keep], s[keep], np.zeros(keep.sum()) if cfg["omit_gamma"] else g[keep]).sum(axis=1)
    scale = core.std()
    hpl = core + rng.normal(0.0, cfg["hpl_noise"] * scale, len(core))
    rtpl = rt + rng.normal(0.0, cfg["rtpl_noise"] * scale, len(core))
    return hpl, rtpl


def main() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inp = build_inputs(BOOK_SEED, PANEL_SEED)
    rng = np.random.default_rng(NOISE_SEED)
    rows = []
    for desk, cfg in DESKS.items():
        hpl, rtpl = desk_series(inp, desk, cfg, rng)
        pla = pla_test(hpl, rtpl)
        b99 = B.backtest(hpl, 0.99)
        b975 = B.backtest(hpl, 0.975)
        rows.append({"desk": desk, "spearman": pla.spearman, "ks": pla.ks, "ks_p_value": pla.ks_pvalue,
                     "pla_zone": pla.zone, "exceptions_99": b99.exceptions, "exceptions_975": b975.exceptions,
                     "outcome": desk_eligibility(pla.zone, b99.exceptions, b975.exceptions)})
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "pla_desk_results.csv", index=False)
    with open(OUTPUT_DIR / "pla_desk_results.md", "w", encoding="utf-8") as fh:
        fh.write("# P&L attribution test by desk (synthetic)\n\n" + md_table(df, "{:,.4f}") + "\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    return {"table": df}


if __name__ == "__main__":
    main()
