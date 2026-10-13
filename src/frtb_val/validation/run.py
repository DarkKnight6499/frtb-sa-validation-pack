"""Run the full validation, derive monitoring statuses and severity-rated findings."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import benchmarks as B
from . import outcomes as O
from .conceptual import assumption_register, run_conceptual_checks
from .model import FrontOfficePricer

# (metric, green upper bound, amber upper bound). Above amber is red. Policy choices for this exercise.
MONITORING_THRESHOLDS = {
    "max_abs_z_vs_mc": (3.0, 4.0),
    "max_abs_diff_european_vs_fd": (0.01, 0.03),
    "max_abs_diff_american_tree_vs_fd": (0.01, 0.03),
    "max_parity_residual": (1e-9, 1e-6),
    "max_rel_diff_delta_european": (0.01, 0.03),
    "tree_oscillation_amplitude": (0.005, 0.02),
    "rmse_vol_pts_in_band": (0.5, 1.0),
    "rmse_vol_pts_all_quotes": (1.0, 2.0),
    "hedge_std_over_premium_model_vol": (0.05, 0.10),
    "hedge_std_over_premium_vol_shock": (0.10, 0.20),
}


def status(metric: str, value: float) -> str:
    green, amber = MONITORING_THRESHOLDS[metric]
    return "Green" if value <= green else ("Amber" if value <= amber else "Red")


@dataclass
class ValidationResults:
    model: FrontOfficePricer
    conceptual: list
    assumptions: list
    benchmark: pd.DataFrame
    american: pd.DataFrame
    convergence: dict
    greeks: pd.DataFrame
    pricing_outcomes: pd.DataFrame
    pricing_summary: dict
    hedge_base: dict
    hedge_shock: dict
    hedge_weekly: dict
    monitoring: pd.DataFrame
    findings: pd.DataFrame
    conclusion: str


def build_monitoring(conceptual: list, bench: pd.DataFrame, amer: pd.DataFrame, conv: dict, greeks: pd.DataFrame,
                     summary: dict, hedge_base: dict, hedge_shock: dict) -> pd.DataFrame:
    parity = next(c for c in conceptual if c["check"] == "European put-call parity")
    parity_val = float(parity["detail"].split()[-1])
    eur_delta = greeks[(greeks["model"] == "European call") & (greeks["greek"] == "delta")]
    values = {
        "max_abs_z_vs_mc": float(bench["z_score"].abs().max()),
        "max_abs_diff_european_vs_fd": float(bench["fo_minus_fd"].abs().max()),
        "max_abs_diff_american_tree_vs_fd": float(amer["tree_minus_fd"].abs().max()),
        "max_parity_residual": parity_val,
        "max_rel_diff_delta_european": float(eur_delta["rel_diff"].max()),
        "tree_oscillation_amplitude": float(conv["tree_oscillation"]),
        "rmse_vol_pts_in_band": float(summary["rmse_vol_pts_in_band"]),
        "rmse_vol_pts_all_quotes": float(summary["rmse_vol_pts"]),
        "hedge_std_over_premium_model_vol": float(hedge_base["std_over_premium"]),
        "hedge_std_over_premium_vol_shock": float(hedge_shock["std_over_premium"]),
    }
    return pd.DataFrame([{"metric": m, "value": v, "green_max": MONITORING_THRESHOLDS[m][0],
                          "amber_max": MONITORING_THRESHOLDS[m][1], "status": status(m, v)} for m, v in values.items()])


def _sev(label: str, severity: str, area: str, evidence: str, rec: str) -> dict:
    return {"id": label, "severity": severity, "area": area, "evidence": evidence, "recommendation": rec}


def build_findings(bench: pd.DataFrame, amer: pd.DataFrame, conv: dict, greeks: pd.DataFrame, summary: dict,
                   hedge_base: dict, hedge_shock: dict, conceptual: list) -> pd.DataFrame:
    """Severity rules: High = limit breached with material error; Medium = limit breached or untested scope that
    affects use; Low = observation with small measured impact; Info = test passed, recorded for completeness."""
    f = []
    max_err = summary["max_abs_vol_error_pts"]
    sev = "High" if max_err > 5 else ("Medium" if max_err > 2 else ("Low" if max_err > 0.5 else "Info"))
    if summary["breaches"] > 0 and sev in ("Low", "Info"):
        sev = "Medium"
    f.append(_sev("F1", sev, "Conceptual soundness / outcomes",
                  f"Flat volatility misprices a skewed market: max implied-vol error {max_err:.2f} vol points, "
                  f"RMSE {summary['rmse_vol_pts']:.2f} points over {summary['n_quotes']} quotes; "
                  f"{summary['breaches']} of {summary['n_in_band']} in-band quotes breach the "
                  f"{O.VOL_ERROR_LIMIT:.1f} point limit.",
                  "Restrict use to the moneyness band where the limit holds, or replace the flat surface with a "
                  "strike and maturity dependent volatility input."))
    z = float(bench["z_score"].abs().max())
    d = float(bench["fo_minus_fd"].abs().max())
    f.append(_sev("F2", "Info" if (z < B.Z_LIMIT and d < 0.01) else "Medium", "Developmental evidence",
                  f"European closed form agrees with Monte Carlo (max |z| {z:.2f}, {int(bench['mc_ci_contains'].sum())} "
                  f"of {len(bench)} 95% intervals contain it) and with the PDE (max difference {d:.4f}).",
                  "None. Keep the benchmark in the monitoring pack."))
    ad = float(amer["tree_minus_fd"].abs().max())
    f.append(_sev("F3", "Info" if ad <= 0.01 else ("Low" if ad <= 0.03 else "Medium"), "Developmental evidence",
                  f"American put tree (500 steps) versus PDE challenger: max difference {ad:.4f} "
                  f"(versus 5000-step tree: {float(amer['tree_minus_ref'].abs().max()):.4f}).",
                  "Raise tree steps or add Richardson extrapolation if the tolerance is tightened."))
    osc = conv["tree_oscillation"]
    f.append(_sev("F4", "Low" if osc > 1e-3 else "Info", "Developmental evidence",
                  f"Binomial price oscillates with step parity: spread {osc:.5f} across 200 to 203 steps; "
                  f"measured convergence order {-conv['tree_order']:.2f} (about 1 expected).",
                  "Use an odd-even smoothing (for example Leisen-Reimer or averaging two step counts)."))
    gam = greeks[(greeks["model"] == "European call") & (greeks["greek"] == "gamma")]["rel_diff"].max()
    f.append(_sev("F5", "Low" if gam > 0.05 else "Info", "Developmental evidence",
                  f"European delta agrees with the Monte Carlo challenger within "
                  f"{greeks[(greeks['model'] == 'European call') & (greeks['greek'] == 'delta')]['rel_diff'].max():.2%}; "
                  f"gamma differences up to {gam:.1%} reflect challenger sampling noise, not model error.",
                  "Treat the analytic Greeks as the reference; validate the challenger's gamma with a larger sample."))
    f.append(_sev("F6", "Low" if hedge_shock["std_over_premium"] > O.HEDGE_STD_LIMIT else "Info", "Outcomes analysis",
                  f"Daily delta hedging error is {hedge_base['std_over_premium']:.1%} of premium at the model "
                  f"volatility and {hedge_shock['std_over_premium']:.1%} when realised volatility is "
                  f"{hedge_shock['true_vol']:.0%} against the model's flat 20%; mean error {hedge_shock['mean_error']:.3f}.",
                  "Add a volatility-uncertainty reserve or monitor realised against implied volatility."))
    f.append(_sev("F7", "Medium", "Limitations and assumptions",
                  "Discrete dividends are not supported: the model takes a continuous yield, which misstates early "
                  "exercise value for American options around ex-dividend dates. No numeric test was possible.",
                  "Add a discrete dividend treatment or limit the model to indices and low-dividend underlyings."))
    f.append(_sev("F8", "Low", "Limitations and assumptions",
                  "American calls priced by the front-office tree have no independent challenger: the PDE solver "
                  "implemented here supports American puts only.",
                  "Extend the challenger to calls before approving the model for American calls with dividends."))
    failed = [c["check"] for c in conceptual if not c["passed"]]
    f.append(_sev("F9", "High" if failed else "Info", "Conceptual soundness",
                  f"{len(conceptual) - len(failed)} of {len(conceptual)} conceptual soundness checks passed"
                  + (f"; failed: {', '.join(failed)}." if failed else "."),
                  "Resolve failing checks before use." if failed else "None."))
    return pd.DataFrame(f)


def overall_conclusion(findings: pd.DataFrame) -> str:
    sev = set(findings["severity"])
    if "High" in sev:
        return "Not approved: at least one High finding is open."
    if "Medium" in sev:
        return "Approved with conditions: Medium findings require remediation or a documented use restriction."
    return "Approved: no High or Medium findings."


def run_validation(model: FrontOfficePricer | None = None, mc_paths: int = 200_000, fd_grid: int = 800,
                   hedge_paths: int = 2000) -> ValidationResults:
    model = model or FrontOfficePricer()
    conceptual = run_conceptual_checks(model)
    bench = B.benchmark_grid(model, mc_paths=mc_paths, fd_grid=fd_grid)
    amer = B.american_benchmark(model, fd_grid=fd_grid)
    conv = B.convergence(model)
    greeks = B.greeks_comparison(model, mc_paths=mc_paths, fd_grid=fd_grid)
    po = O.pricing_outcomes(model)
    summary = O.pricing_summary(po)
    hb = O.hedging_backtest(model, None, hedge_paths)
    hs = O.hedging_backtest(model, 0.30, hedge_paths)
    hw = O.hedging_backtest(model, None, hedge_paths, rebalance_every=5)
    mon = build_monitoring(conceptual, bench, amer, conv, greeks, summary, hb, hs)
    findings = build_findings(bench, amer, conv, greeks, summary, hb, hs, conceptual)
    return ValidationResults(model, conceptual, assumption_register(), bench, amer, conv, greeks, po, summary, hb, hs,
                             hw, mon, findings, overall_conclusion(findings))
