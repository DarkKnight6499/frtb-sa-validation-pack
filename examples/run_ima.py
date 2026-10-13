"""IMA expected shortfall, stressed ES, NMRF charge and backtesting on the synthetic book."""
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
N_DAYS = 2500
STRESS_START = 700            # planted stress window start (day index), 250 days long
OBS_SEED = 5                  # real price observation generator
BACKTEST_ALPHAS = (0.99, 0.975)
WEAK_VAR_SCALE = 0.7          # deliberately understated VaR, to show the traffic light reacting
# ==
sys.path.insert(0, str(ROOT / "src"))

from frtb_val import columns as C  # noqa: E402
from frtb_val.ima import backtest as B  # noqa: E402
from frtb_val.ima import es as E  # noqa: E402
from frtb_val.ima import factors as F  # noqa: E402
from frtb_val.ima import nmrf as N  # noqa: E402
from frtb_val.ima.pipeline import build_inputs  # noqa: E402
from frtb_val.validation.report import md_table  # noqa: E402


def main() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inp = build_inputs(BOOK_SEED, PANEL_SEED, N_DAYS, STRESS_START)
    meta, lh, broad = inp.meta, inp.lh, inp.broad

    # Risk factor eligibility: factors failing the RFET leave the ES model and receive a stress scenario charge.
    obs = N.synthetic_observations(meta, OBS_SEED)
    rfet_rows = []
    for _, row in meta.iterrows():
        res = N.rfet(obs[row[F.FACTOR_ID]])
        rfet_rows.append({F.FACTOR_ID: row[F.FACTOR_ID], F.LIQ_HORIZON: row[F.LIQ_HORIZON], **res})
    rfet_df = pd.DataFrame(rfet_rows)
    modellable = rfet_df["modellable"].to_numpy()
    rfet_df.to_csv(OUTPUT_DIR / "ima_rfet.csv", index=False)

    cur = slice(inp.contrib10.shape[0] - E.WINDOW, inp.contrib10.shape[0])
    idx_m = np.where(modellable)[0]
    reduced_local = E.select_reduced_set(inp.contrib10[cur][:, idx_m], lh[idx_m], broad[idx_m])
    ima = E.stressed_es(inp.contrib10[:, idx_m], lh[idx_m], reduced_local, broad[idx_m])

    # Plain ES against the liquidity horizon cascade on the current window, modellable factors.
    cur10 = inp.contrib10[cur][:, idx_m]
    plain10 = E.cascade_es(cur10, np.full(len(idx_m), 10))
    cascade = E.cascade_es(cur10, lh[idx_m])
    conf_rows = [{"alpha": a, "ES_10d_plain": E.expected_shortfall(cur10.sum(axis=1), a)}
                 for a in (0.90, 0.95, 0.975, 0.99)]

    # NMRF stress scenario charges over the planted-stress 12-month window of daily changes.
    stress_days = slice(STRESS_START, STRESS_START + 250)
    nm = np.where(~modellable)[0]
    ses_rows = []
    for i in nm:
        row = meta.iloc[i]
        x = inp.panel.iloc[:, i].to_numpy()[stress_days]
        ses_rows.append({F.FACTOR_ID: row[F.FACTOR_ID], F.LIQ_HORIZON: row[F.LIQ_HORIZON],
                         "ses_k": N.stress_scenario_charge(x, row[C.SENSITIVITY], row[C.GAMMA], row[F.LIQ_HORIZON])})
    ses_df = pd.DataFrame(ses_rows, columns=[F.FACTOR_ID, F.LIQ_HORIZON, "ses_k"])
    ses_total = N.ses(ses_df["ses_k"].to_numpy()) if len(ses_df) else 0.0
    ses_df.to_csv(OUTPUT_DIR / "ima_nmrf_ses.csv", index=False)

    # Backtesting of the full-book one-day P&L (hypothetical P&L, static positions).
    pnl = inp.daily.sum(axis=1)
    bt_rows = []
    for a in BACKTEST_ALPHAS:
        for label, scale in (("model VaR", 1.0), (f"VaR scaled to {WEAK_VAR_SCALE:.0%}", WEAK_VAR_SCALE)):
            r = B.backtest(pnl, a, scale=scale)
            test = B.binomial_exceedance_test(r.exceptions, r.n, a)
            kup = B.kupiec_pof(r.exceptions, r.n, a)
            bt_rows.append({"confidence": a, "var": label, "n": r.n, "exceptions": r.exceptions,
                            "expected": round(test["expected"], 2), "zone": r.zone,
                            "p_too_many": test["p_too_many"], "kupiec_p": kup["p_value"]})
    bt_df = pd.DataFrame(bt_rows)
    bt_df.to_csv(OUTPUT_DIR / "ima_backtesting.csv", index=False)

    amber, red = B.zone_boundaries(250, 0.99)
    summary = pd.DataFrame([
        {"item": "risk factors total", "value": len(meta)},
        {"item": "risk factors failing RFET (NMRF)", "value": int((~modellable).sum())},
        {"item": "reduced set size (modellable)", "value": len(reduced_local)},
        {"item": "ES_F,C current, full set (USD)", "value": ima.es_fc},
        {"item": "ES_R,C current, reduced set (USD)", "value": ima.es_rc},
        {"item": "ES_R,C / ES_F,C (needs >= 0.75)", "value": ima.explained},
        {"item": "ES_R,S stressed, reduced set (USD)", "value": ima.es_rs},
        {"item": "stressed window start scenario (planted at day 700)", "value": ima.stress_start},
        {"item": "ES = ES_R,S x max(ES_F,C / ES_R,C, 1) (USD)", "value": ima.es_stressed},
        {"item": "IMCC with rho = 0.5 (USD)", "value": ima.imcc},
        {"item": "SES (USD)", "value": ses_total},
        {"item": "IMCC + SES (USD)", "value": ima.imcc + ses_total},
        {"item": "plain 10-day ES, all horizons forced to 10 (USD)", "value": plain10},
        {"item": "liquidity-adjusted cascade ES, current window (USD)", "value": cascade},
        {"item": "amber zone starts at (250 obs, 99%)", "value": amber},
        {"item": "red zone starts at (250 obs, 99%)", "value": red},
    ])
    summary.to_csv(OUTPUT_DIR / "ima_summary.csv", index=False)
    with open(OUTPUT_DIR / "ima_summary.md", "w", encoding="utf-8") as fh:
        fh.write("# IMA results (synthetic book and synthetic factor panel)\n\n" + md_table(summary, "{:,.4f}") + "\n\n")
        fh.write("ES by broad risk class (stressed, constrained):\n\n")
        fh.write(md_table(pd.DataFrame([{"class": k, "ES": v} for k, v in ima.by_class.items()]), "{:,.0f}") + "\n\n")
        fh.write("Plain 10-day ES by confidence level:\n\n" + md_table(pd.DataFrame(conf_rows), "{:,.0f}") + "\n\n")
        fh.write("Backtesting, last 250 days:\n\n" + md_table(bt_df, "{:,.4f}") + "\n\n")
        fh.write("Factors failing the RFET:\n\n"
                 + md_table(rfet_df[~rfet_df["modellable"]][[F.FACTOR_ID, F.LIQ_HORIZON, "n_obs", "min_obs_in_90d",
                                                              "criterion_1", "criterion_2"]], "{:,.0f}") + "\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print(bt_df.to_string(index=False))
    return {"ima": ima, "summary": summary, "backtest": bt_df, "rfet": rfet_df, "ses_total": ses_total,
            "plain10": plain10, "cascade": cascade, "conf_rows": conf_rows}


if __name__ == "__main__":
    main()
