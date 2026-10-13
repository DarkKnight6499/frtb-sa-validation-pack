"""FRTB standardised approach on the seeded synthetic book. Writes CSV and Markdown under examples/output/."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# == config
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "examples" / "output"
BOOK_SEED = 7
# ==
sys.path.insert(0, str(ROOT / "src"))

from frtb_val import columns as C  # noqa: E402
from frtb_val.frtb_sa.capital import sa_capital  # noqa: E402
from frtb_val.frtb_sa.sbm import net_sensitivities  # noqa: E402
from frtb_val.portfolio import (book_value, curvature, default_book_and_market, drc_positions,  # noqa: E402
                                rrao_positions, sensitivities)
from frtb_val.validation.report import md_table  # noqa: E402


def main() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    book, market = default_book_and_market(BOOK_SEED)
    sens = sensitivities(book, market)
    curv = curvature(book, market)
    drc_pos = drc_positions(book, market)
    rrao_pos = rrao_positions(book, market)
    res = sa_capital(sens, curv, drc_pos, rrao_pos)

    sens.to_csv(OUTPUT_DIR / "sa_sensitivities_by_instrument.csv", index=False)
    net_sensitivities(sens).to_csv(OUTPUT_DIR / "sa_net_sensitivities.csv", index=False)
    curv.to_csv(OUTPUT_DIR / "sa_curvature_positions.csv", index=False)
    res.decomposition.to_csv(OUTPUT_DIR / "sa_capital_decomposition.csv", index=False)
    res.drc.by_bucket.to_csv(OUTPUT_DIR / "sa_drc_by_bucket.csv", index=False)

    dec = res.decomposition.copy()
    scen_cols = [c for c in dec.columns if c.endswith("_scenario")]
    with open(OUTPUT_DIR / "sa_capital_decomposition.md", "w", encoding="utf-8") as fh:
        fh.write("# Standardised approach capital (synthetic book)\n\n")
        fh.write(f"Book of {len(book)} instruments, net market value {book_value(book, market):,.0f} USD. ")
        fh.write(f"Binding SBM correlation scenario: **{res.sbm.selected}**.\n\n")
        fh.write(md_table(dec, "{:,.0f}") + "\n\n")
        fh.write("SBM total by scenario:\n\n")
        fh.write(md_table(pd.DataFrame(
            [{"scenario": k, "SBM total": v} for k, v in res.sbm.totals.items()]), "{:,.0f}") + "\n\n")
        fh.write("Default risk charge by bucket:\n\n" + md_table(res.drc.by_bucket, "{:,.4f}") + "\n")
    print(dec[[C.RISK_CLASS, C.MEASURE, "charge"] + scen_cols].to_string(index=False, float_format=lambda x: f"{x:,.0f}"))
    print("SBM totals:", {k: round(v) for k, v in res.sbm.totals.items()}, "binding:", res.sbm.selected)
    print(f"DRC {res.drc.total:,.0f}  RRAO {res.rrao:,.0f}  TOTAL {res.total:,.0f}")
    return {"result": res, "book": book, "market": market}


if __name__ == "__main__":
    main()
