"""Markdown rendering helpers and the SR 11-7 structured validation report."""
from __future__ import annotations

import pandas as pd

from . import benchmarks as B
from . import outcomes as O
from .run import MONITORING_THRESHOLDS, ValidationResults


def md_table(df: pd.DataFrame, floatfmt: str = "{:,.4f}", formats: dict | None = None) -> str:
    """Render a DataFrame as a GitHub markdown table without extra dependencies."""
    formats = formats or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in formats:
                cells.append(formats[c].format(v))
            elif isinstance(v, float) and v != v:
                cells.append("")
            elif isinstance(v, bool):
                cells.append("yes" if v else "no")
            elif isinstance(v, float):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_report(r: ValidationResults) -> str:
    m, bench, amer, conv = r.model, r.benchmark, r.american, r.convergence
    s = r.pricing_summary
    out: list[str] = []
    w = out.append
    w("# Independent Model Validation Report: Vanilla Option Pricer")
    w("")
    w("Structured after the headings of Supervisory Guidance on Model Risk Management (SR 11-7, Board of Governors of the "
      "Federal Reserve System and OCC, April 2011): Model Development, Implementation, and Use; Model Validation "
      "(Evaluation of Conceptual Soundness, Ongoing Monitoring, Outcomes Analysis); Governance. All figures below were "
      "produced by `examples/run_validation.py` on synthetic inputs with fixed seeds.")
    w("")
    w("## 1. Executive Summary")
    w("")
    w(f"**Conclusion: {r.conclusion}**")
    w("")
    counts = r.findings["severity"].value_counts().to_dict()
    w("Findings by severity: " + ", ".join(f"{k} {counts.get(k, 0)}" for k in ("High", "Medium", "Low", "Info")) + ".")
    w("")
    w(f"The closed-form European prices agree with the Monte Carlo challenger (max |z| {bench['z_score'].abs().max():.2f}) "
      f"and the finite-difference challenger (max difference {bench['fo_minus_fd'].abs().max():.4f}). The American put tree "
      f"agrees with the PDE challenger to {amer['tree_minus_fd'].abs().max():.4f}. The model's main weakness is the flat "
      f"volatility assumption: against a synthetic skewed market the implied-vol error reaches "
      f"{s['max_abs_vol_error_pts']:.2f} vol points and {s['breaches']} of {s['n_in_band']} in-band quotes breach the "
      f"{O.VOL_ERROR_LIMIT:.1f} point limit.")
    w("")
    w("## 2. Model Description")
    w("")
    w(f"Model under validation: a front-office pricer with one flat volatility ({m.sigma:.0%}), a constant risk-free rate "
      f"({m.r:.1%}) and a continuous dividend yield ({m.q:.1%}).")
    w("")
    w("- European calls and puts: Black-Scholes-Merton closed form.")
    w(f"- American puts: Cox-Ross-Rubinstein binomial tree with {m.tree_steps} steps and early exercise at every node.")
    w("")
    w("Challengers built independently for this validation:")
    w("")
    w(f"- Monte Carlo with antithetic draws and a regression control variate (the discounted terminal price), "
      f"{B.MC_PATHS:,} paths.")
    w(f"- Crank-Nicolson finite differences in log price with Rannacher start-up and Brennan-Schwartz projection for "
      f"early exercise, {B.FD_GRID} space and {B.FD_GRID} time steps.")
    w("")
    w("Intended use is not documented for this synthetic model, so every strike and maturity on the test grid is treated "
      "as in scope.")
    w("")
    w("## 3. Conceptual Soundness")
    w("")
    w("SR 11-7 asks for an evaluation of design, theory and assumptions. Property checks:")
    w("")
    w(md_table(pd.DataFrame(r.conceptual).rename(columns={"check": "Check", "passed": "Passed", "detail": "Detail"})))
    w("")
    w("Assumption register:")
    w("")
    w(md_table(pd.DataFrame(r.assumptions).rename(columns={"assumption": "Assumption", "assessment": "Assessment"})))
    w("")
    w("## 4. Developmental Evidence")
    w("")
    w("### 4.1 Benchmark grid (strikes 80 to 120, maturities 0.25 to 2 years, spot 100)")
    w("")
    w(f"European prices: {len(bench)} price comparisons. Monte Carlo z-score is (front office minus challenger) over the "
      f"challenger standard error; the variance reduction from the control variate averages "
      f"{bench['variance_reduction'].mean():.1f}x.")
    w("")
    sub = bench[["type", "strike", "maturity", "front_office", "mc_cv", "mc_stderr", "z_score", "fd", "fo_minus_fd"]]
    w(md_table(sub, "{:,.4f}", {"strike": "{:.0f}", "maturity": "{:.2f}"}))
    w("")
    w("American put, tree against PDE:")
    w("")
    w(md_table(amer, "{:,.4f}", {"strike": "{:.0f}", "maturity": "{:.2f}"}))
    w("")
    w("### 4.2 Convergence")
    w("")
    w(f"Binomial tree, ATM one-year European put against the closed form (fitted log-log slope "
      f"{conv['tree_order']:.2f}; step-parity oscillation {conv['tree_oscillation']:.5f}):")
    w("")
    w(md_table(conv["tree"], "{:,.6f}", {"steps": "{:.0f}"}))
    w("")
    w(f"Monte Carlo, standard error against paths (fitted slope {conv['mc_order']:.2f}, about -0.5 expected):")
    w("")
    w(md_table(conv["mc"], "{:,.6f}", {"paths": "{:.0f}"}))
    w("")
    w(f"Finite-difference grid, American put against an 8000-step tree ({conv['american_reference']:.6f}); fitted slope "
      f"over the first five grids {conv['fd_order']:.2f}:")
    w("")
    w(md_table(conv["fd"], "{:,.6f}", {"grid": "{:.0f}"}))
    w("")
    w("### 4.3 Greeks comparison")
    w("")
    w(md_table(r.greeks, "{:,.5f}", {"strike": "{:.0f}", "maturity": "{:.2f}"}))
    w("")
    w("## 5. Ongoing Monitoring")
    w("")
    w("Metrics, thresholds and current status. Thresholds are policy choices set for this exercise; Green is at or below "
      "the green bound, Amber at or below the amber bound, Red above it.")
    w("")
    mon = r.monitoring.copy()
    w(md_table(mon, "{:,.6g}"))
    w("")
    w(f"Statuses: {(mon['status'] == 'Green').sum()} Green, {(mon['status'] == 'Amber').sum()} Amber, "
      f"{(mon['status'] == 'Red').sum()} Red. Re-run the pack on every model change and at least quarterly.")
    w("")
    w("## 6. Outcomes Analysis")
    w("")
    w("### 6.1 Model prices against a synthetic smile market")
    w("")
    w("The market is generated from a stated smile, `sigma(K, T) = 0.20 + 0.02 exp(-T) - 0.15 m + 0.40 m^2` with "
      "`m = ln(K/F)`, plus 0.2 vol point of quote noise. Out-of-the-money options are quoted. The model reprices each "
      "quote at its flat 20% volatility and the table gives the implied-vol error.")
    w("")
    po = r.pricing_outcomes[["type", "strike", "maturity", "market_vol", "model_vol", "vol_error_pts", "market_price",
                             "model_price", "in_band", "breach"]]
    w(md_table(po, "{:,.4f}", {"strike": "{:.0f}", "maturity": "{:.2f}"}))
    w("")
    w(f"Limit test: inside the band |ln(K/F)| <= {O.MONEYNESS_BAND:.2f} the absolute vol error must stay within "
      f"{O.VOL_ERROR_LIMIT:.1f} point for at least {1 - O.BREACH_RATE_LIMIT:.0%} of quotes. Result: {s['breaches']} "
      f"breaches in {s['n_in_band']} in-band quotes ({s['breach_rate']:.0%}), RMSE {s['rmse_vol_pts_in_band']:.2f} points in "
      f"band and {s['rmse_vol_pts']:.2f} over all quotes. **Limit {'passed' if s['limit_passed'] else 'failed'}.**")
    w("")
    w("### 6.2 Delta-hedging backtest")
    w("")
    w("One-year at-the-money call sold at the model price and delta hedged at the model volatility, 250 steps, "
      f"{r.hedge_base['n_paths']:,} simulated paths. Error is terminal hedge value minus payoff.")
    w("")
    hedge = pd.DataFrame([
        {"scenario": "realised vol = model vol, daily hedge", **{k: r.hedge_base[k] for k in
         ("mean_error", "std_error", "std_over_premium", "p01_error", "p99_error")}, "limit_passed": r.hedge_base["limit_passed"]},
        {"scenario": "realised vol = model vol, weekly hedge", **{k: r.hedge_weekly[k] for k in
         ("mean_error", "std_error", "std_over_premium", "p01_error", "p99_error")}, "limit_passed": r.hedge_weekly["limit_passed"]},
        {"scenario": "realised vol 30%, daily hedge", **{k: r.hedge_shock[k] for k in
         ("mean_error", "std_error", "std_over_premium", "p01_error", "p99_error")}, "limit_passed": r.hedge_shock["limit_passed"]},
    ])
    w(md_table(hedge, "{:,.4f}"))
    w("")
    w(f"Premium is {r.hedge_base['premium']:.4f}. Limit: standard deviation of error within {O.HEDGE_STD_LIMIT:.0%} of premium.")
    w("")
    w("## 7. Limitations and Assumptions")
    w("")
    w("- The market data in section 6 is synthetic; the smile is a stated function, not an observation.")
    w("- Monte Carlo results carry sampling error; the challenger Greeks use common random numbers and finite differences.")
    w("- The PDE challenger covers European options and American puts. American calls and discrete dividends are untested.")
    w("- Rates and dividend yield are constant; no stochastic volatility, jumps or funding effects are modelled or tested.")
    w("- Monitoring thresholds are exercise policy, not regulatory values.")
    w("")
    w("## 8. Findings and Recommendations")
    w("")
    w("Severity scale: High (limit breached with material error, or a conceptual check fails), Medium (limit breached or "
      "untested scope that affects use), Low (observation with small measured impact), Info (test passed, recorded).")
    w("")
    w(md_table(r.findings.rename(columns={"id": "ID", "severity": "Severity", "area": "Area", "evidence": "Evidence",
                                           "recommendation": "Recommendation"})))
    w("")
    w("## 9. Conclusion")
    w("")
    w(r.conclusion)
    w("")
    high = r.findings[r.findings["severity"] == "High"]["id"].tolist()
    med = r.findings[r.findings["severity"] == "Medium"]["id"].tolist()
    w(f"Open High findings: {', '.join(high) if high else 'none'}. Open Medium findings: {', '.join(med) if med else 'none'}.")
    w("")
    return "\n".join(out)
