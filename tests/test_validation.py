"""Tests for the pricers, challengers, outcomes analysis, findings and the SR 11-7 report."""
import math
import re
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frtb_val.pricing import (bs_greeks, bs_implied_vol, bs_price, crr_binomial, crr_greeks)  # noqa: E402
from frtb_val.validation import outcomes as O  # noqa: E402
from frtb_val.validation.challenger import fd_option, mc_european, mc_greeks_crn  # noqa: E402
from frtb_val.validation.conceptual import run_conceptual_checks  # noqa: E402
from frtb_val.validation.model import FrontOfficePricer  # noqa: E402
from frtb_val.validation.report import md_table, render_report  # noqa: E402
from frtb_val.validation.run import (MONITORING_THRESHOLDS, build_findings, overall_conclusion,  # noqa: E402
                                     run_validation, status)

R, Q, SIG = 0.05, 0.02, 0.20


class BlackScholes(unittest.TestCase):
    def test_textbook_values(self):
        self.assertAlmostEqual(bs_price(100, 100, 0.05, 0.0, 0.2, 1.0, True), 10.450583572185565, places=9)
        self.assertAlmostEqual(bs_price(100, 100, 0.05, 0.0, 0.2, 1.0, False), 5.573526022256971, places=9)

    def test_put_call_parity(self):
        for k in (70, 100, 130):
            for t in (0.1, 1.0, 3.0):
                c, p = bs_price(100, k, R, Q, SIG, t, True), bs_price(100, k, R, Q, SIG, t, False)
                self.assertAlmostEqual(c - p, 100 * math.exp(-Q * t) - k * math.exp(-R * t), places=10)

    def test_greeks_against_finite_differences(self):
        s, k, t = 100.0, 105.0, 0.75
        g = bs_greeks(s, k, R, Q, SIG, t, True)
        h = 1e-3
        fd_delta = (bs_price(s + h, k, R, Q, SIG, t) - bs_price(s - h, k, R, Q, SIG, t)) / (2 * h)
        fd_gamma = (bs_price(s + h, k, R, Q, SIG, t) - 2 * bs_price(s, k, R, Q, SIG, t) + bs_price(s - h, k, R, Q, SIG, t)) / h ** 2
        fd_vega = (bs_price(s, k, R, Q, SIG + h, t) - bs_price(s, k, R, Q, SIG - h, t)) / (2 * h)
        fd_rho = (bs_price(s, k, R + h, Q, SIG, t) - bs_price(s, k, R - h, Q, SIG, t)) / (2 * h)
        fd_theta = -(bs_price(s, k, R, Q, SIG, t + h) - bs_price(s, k, R, Q, SIG, t - h)) / (2 * h)
        self.assertAlmostEqual(g["delta"], fd_delta, places=6)
        self.assertAlmostEqual(g["gamma"], fd_gamma, places=5)
        self.assertAlmostEqual(g["vega"], fd_vega, places=4)
        self.assertAlmostEqual(g["rho"], fd_rho, places=4)
        self.assertAlmostEqual(g["theta"], fd_theta, places=5)

    def test_implied_vol_round_trip(self):
        p = bs_price(100, 90, R, Q, 0.37, 0.5, False)
        self.assertAlmostEqual(bs_implied_vol(p, 100, 90, R, Q, 0.5, False), 0.37, places=9)

    def test_degenerate_inputs_return_intrinsic(self):
        self.assertAlmostEqual(bs_price(120, 100, 0.0, 0.0, 0.2, 0.0, True), 20.0)
        self.assertEqual(bs_price(80, 100, 0.0, 0.0, 0.2, 0.0, True), 0.0)

    def test_monotone_in_volatility_and_maturity(self):
        v = [bs_price(100, 100, R, Q, s, 1.0) for s in (0.1, 0.2, 0.3)]
        t = [bs_price(100, 100, R, Q, SIG, x) for x in (0.25, 0.5, 1.0)]
        self.assertTrue(v[0] < v[1] < v[2] and t[0] < t[1] < t[2])


class Binomial(unittest.TestCase):
    def test_european_binomial_converges_to_black_scholes(self):
        exact = bs_price(100, 100, R, Q, SIG, 1.0, False)
        errs = [abs(crr_binomial(100, 100, R, Q, SIG, 1.0, n, False) - exact) for n in (50, 200, 800, 3200)]
        self.assertLess(errs[-1], 1e-3)
        self.assertLess(errs[-1], errs[0] / 10)

    def test_call_converges_too(self):
        exact = bs_price(100, 110, R, Q, SIG, 2.0, True)
        self.assertAlmostEqual(crr_binomial(100, 110, R, Q, SIG, 2.0, 3000, True), exact, delta=1e-3)

    def test_american_put_known_value(self):
        self.assertAlmostEqual(crr_binomial(100, 100, 0.05, 0.0, 0.2, 1.0, 4000, False, True), 6.0903, delta=2e-3)

    def test_american_put_dominates_european_and_intrinsic(self):
        a = crr_binomial(80, 100, R, Q, SIG, 1.0, 500, False, True)
        e = crr_binomial(80, 100, R, Q, SIG, 1.0, 500, False, False)
        self.assertGreaterEqual(a, e)
        self.assertGreaterEqual(a, 20.0)

    def test_american_call_equals_european_without_dividends(self):
        a = crr_binomial(100, 95, R, 0.0, SIG, 1.0, 400, True, True)
        e = crr_binomial(100, 95, R, 0.0, SIG, 1.0, 400, True, False)
        self.assertAlmostEqual(a, e, places=10)

    def test_american_call_exceeds_european_with_dividends(self):
        a = crr_binomial(100, 80, R, 0.08, SIG, 2.0, 500, True, True)
        e = crr_binomial(100, 80, R, 0.08, SIG, 2.0, 500, True, False)
        self.assertGreater(a, e + 1e-3)

    def test_tree_greeks_close_to_analytic_for_european(self):
        g = crr_greeks(100, 100, R, Q, SIG, 1.0, 1000, True, False)
        a = bs_greeks(100, 100, R, Q, SIG, 1.0, True)
        self.assertAlmostEqual(g["delta"], a["delta"], delta=2e-3)
        self.assertAlmostEqual(g["gamma"], a["gamma"], delta=2e-4)


class Challengers(unittest.TestCase):
    def test_monte_carlo_within_confidence_interval_of_closed_form(self):
        exact = bs_price(100, 105, R, Q, SIG, 1.0, True)
        res = mc_european(100, 105, R, Q, SIG, 1.0, True, 100_000, seed=11)
        lo, hi = res.ci95
        self.assertTrue(abs(res.price - exact) < 3 * res.stderr)
        self.assertGreater(hi - lo, 0)

    def test_control_variate_reduces_variance(self):
        res = mc_european(100, 100, R, Q, SIG, 1.0, True, 50_000, seed=5)
        self.assertGreater(res.variance_reduction, 3.0)
        self.assertLess(res.stderr, res.plain_stderr)

    def test_monte_carlo_is_unbiased_across_seeds(self):
        exact = bs_price(100, 95, R, Q, SIG, 0.5, False)
        zs = [(mc_european(100, 95, R, Q, SIG, 0.5, False, 20_000, seed=s).price - exact)
              / mc_european(100, 95, R, Q, SIG, 0.5, False, 20_000, seed=s).stderr for s in range(20)]
        self.assertLess(abs(np.mean(zs)), 1.0)
        self.assertLess(max(abs(z) for z in zs), 4.0)

    def test_monte_carlo_standard_error_shrinks_like_root_n(self):
        a = mc_european(100, 100, R, Q, SIG, 1.0, True, 10_000, seed=3).stderr
        b = mc_european(100, 100, R, Q, SIG, 1.0, True, 160_000, seed=3).stderr
        self.assertAlmostEqual(a / b, 4.0, delta=1.0)

    def test_monte_carlo_put_call_parity_with_common_random_numbers(self):
        z = np.random.default_rng(8).standard_normal(50_000)
        c = mc_european(100, 100, R, Q, SIG, 1.0, True, z=z).price
        p = mc_european(100, 100, R, Q, SIG, 1.0, False, z=z).price
        self.assertAlmostEqual(c - p, 100 * math.exp(-Q) - 100 * math.exp(-R), delta=0.05)

    def test_finite_difference_european_matches_closed_form(self):
        for call in (True, False):
            fd = fd_option(100, 100, R, Q, SIG, 1.0, call, False, 400, 400)["price"]
            self.assertAlmostEqual(fd, bs_price(100, 100, R, Q, SIG, 1.0, call), delta=2e-3)

    def test_finite_difference_american_put_matches_tree(self):
        fd = fd_option(100, 100, R, Q, SIG, 1.0, False, True, 400, 400)["price"]
        tree = crr_binomial(100, 100, R, Q, SIG, 1.0, 4000, False, True)
        self.assertAlmostEqual(fd, tree, delta=5e-3)

    def test_finite_difference_american_put_deep_in_the_money_is_intrinsic(self):
        fd = fd_option(50, 100, 0.08, 0.0, 0.2, 2.0, False, True, 400, 400)["price"]
        self.assertAlmostEqual(fd, 50.0, delta=1e-3)

    def test_finite_difference_greeks_close_to_analytic(self):
        fd = fd_option(100, 100, R, Q, SIG, 1.0, True, False, 600, 600)
        g = bs_greeks(100, 100, R, Q, SIG, 1.0, True)
        self.assertAlmostEqual(fd["delta"], g["delta"], delta=2e-3)
        self.assertAlmostEqual(fd["gamma"], g["gamma"], delta=2e-4)

    def test_finite_difference_refinement_reduces_error(self):
        exact = bs_price(100, 100, R, Q, SIG, 1.0, False)
        e1 = abs(fd_option(100, 100, R, Q, SIG, 1.0, False, False, 100, 100)["price"] - exact)
        e2 = abs(fd_option(100, 100, R, Q, SIG, 1.0, False, False, 400, 400)["price"] - exact)
        self.assertLess(e2, e1 / 4)

    def test_american_call_rejected_by_pde_challenger(self):
        with self.assertRaises(ValueError):
            fd_option(100, 100, R, Q, SIG, 1.0, True, True)

    def test_monte_carlo_greeks_close_to_analytic(self):
        g = mc_greeks_crn(100, 100, R, Q, SIG, 1.0, True, 100_000)
        a = bs_greeks(100, 100, R, Q, SIG, 1.0, True)
        self.assertAlmostEqual(g["delta"], a["delta"], delta=5e-3)
        self.assertAlmostEqual(g["vega"], a["vega"], delta=0.5)


class Outcomes(unittest.TestCase):
    def test_flat_vol_matches_flat_market_exactly(self):
        m = FrontOfficePricer()
        # Quote noise aside, a model priced at its own vol must have zero implied-vol error.
        price = m.price(100, 100, 1.0, True)
        self.assertAlmostEqual(bs_implied_vol(price, 100, 100, m.r, m.q, 1.0, True), m.sigma, places=9)

    def test_smile_creates_wing_errors_larger_than_atm(self):
        df = O.pricing_outcomes(FrontOfficePricer())
        wing = df[df["strike"].isin([80.0])]["vol_error_pts"].abs().mean()
        atm = df[df["strike"] == 110.0]["vol_error_pts"].abs().mean()
        self.assertGreater(wing, 3 * atm)

    def test_summary_counts_are_consistent(self):
        df = O.pricing_outcomes(FrontOfficePricer())
        s = O.pricing_summary(df)
        self.assertEqual(s["n_quotes"], 20)
        self.assertEqual(s["breaches"], int(df["breach"].sum()))
        self.assertLessEqual(s["breaches"], s["n_in_band"])

    def test_hedging_error_is_small_and_unbiased_at_model_vol(self):
        h = O.hedging_backtest(FrontOfficePricer(), None, 600, seed=3)
        self.assertLess(abs(h["mean_error"]) / h["premium"], 0.03)
        self.assertLess(h["std_over_premium"], 0.10)

    def test_hedging_error_grows_with_vol_misspecification_and_coarser_hedging(self):
        m = FrontOfficePricer()
        base = O.hedging_backtest(m, None, 600, seed=3)
        shock = O.hedging_backtest(m, 0.30, 600, seed=3)
        weekly = O.hedging_backtest(m, None, 600, rebalance_every=5, seed=3)
        self.assertGreater(shock["std_error"], base["std_error"])
        self.assertLess(shock["mean_error"], base["mean_error"])      # the seller of the option loses
        self.assertGreater(weekly["std_error"], base["std_error"])


class ConceptualChecks(unittest.TestCase):
    def test_all_checks_pass_for_the_default_model(self):
        checks = run_conceptual_checks(FrontOfficePricer())
        failed = [c["check"] for c in checks if not c["passed"]]
        self.assertEqual(failed, [])
        self.assertGreaterEqual(len(checks), 10)


class MonitoringAndFindings(unittest.TestCase):
    def test_status_thresholds(self):
        green, amber = MONITORING_THRESHOLDS["max_abs_z_vs_mc"]
        self.assertEqual(status("max_abs_z_vs_mc", green), "Green")
        self.assertEqual(status("max_abs_z_vs_mc", (green + amber) / 2), "Amber")
        self.assertEqual(status("max_abs_z_vs_mc", amber + 0.1), "Red")

    def test_conclusion_follows_severity(self):
        import pandas as pd
        self.assertIn("Not approved", overall_conclusion(pd.DataFrame({"severity": ["Low", "High"]})))
        self.assertIn("conditions", overall_conclusion(pd.DataFrame({"severity": ["Info", "Medium"]})))
        self.assertTrue(overall_conclusion(pd.DataFrame({"severity": ["Info", "Low"]})).startswith("Approved:"))


class FullValidationRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = run_validation(mc_paths=30_000, fd_grid=200, hedge_paths=300)
        cls.report = render_report(cls.res)

    def test_benchmark_grid_is_complete(self):
        self.assertEqual(len(self.res.benchmark), 40)
        self.assertEqual(len(self.res.american), 20)

    def test_european_prices_agree_with_challengers(self):
        self.assertLess(self.res.benchmark["z_score"].abs().max(), 3.5)
        self.assertLess(self.res.benchmark["fo_minus_fd"].abs().max(), 0.02)

    def test_american_tree_agrees_with_pde(self):
        self.assertLess(self.res.american["tree_minus_fd"].abs().max(), 0.03)

    def test_convergence_orders(self):
        c = self.res.convergence
        self.assertAlmostEqual(c["tree_order"], -1.0, delta=0.3)
        self.assertAlmostEqual(c["mc_order"], -0.5, delta=0.2)
        self.assertLess(c["fd_order"], -1.5)

    def test_findings_table_structure_and_severities(self):
        f = self.res.findings
        self.assertEqual(list(f.columns), ["id", "severity", "area", "evidence", "recommendation"])
        self.assertTrue(set(f["severity"]) <= {"High", "Medium", "Low", "Info"})
        self.assertEqual(f["id"].is_unique, True)
        self.assertTrue((f["evidence"].str.len() > 20).all())

    def test_flat_vol_finding_is_the_high_severity_item(self):
        f = self.res.findings.set_index("id")
        self.assertEqual(f.loc["F1", "severity"], "High")

    def test_monitoring_table_has_every_metric_and_valid_status(self):
        m = self.res.monitoring
        self.assertEqual(set(m["metric"]), set(MONITORING_THRESHOLDS))
        self.assertTrue(set(m["status"]) <= {"Green", "Amber", "Red"})

    def test_report_has_sr_11_7_headings_in_order(self):
        heads = re.findall(r"^## (\d+)\. (.+)$", self.report, flags=re.M)
        self.assertEqual([h[1] for h in heads], ["Executive Summary", "Model Description", "Conceptual Soundness",
                                                 "Developmental Evidence", "Ongoing Monitoring", "Outcomes Analysis",
                                                 "Limitations and Assumptions", "Findings and Recommendations",
                                                 "Conclusion"])

    def test_report_contains_computed_numbers_not_placeholders(self):
        self.assertNotIn("TODO", self.report)
        self.assertIsNone(re.search(r"nan", self.report.lower()))
        self.assertIn(f"{self.res.benchmark['z_score'].abs().max():.2f}", self.report)
        self.assertIn(self.res.conclusion, self.report)

    def test_report_has_no_dash_substitutes(self):
        self.assertNotIn(chr(0x2014), self.report)
        self.assertNotIn(chr(0x2013), self.report)
        body = chr(10).join(x for x in self.report.splitlines() if not x.startswith("|---"))
        self.assertNotIn("--", body)

    def test_markdown_table_renderer(self):
        import pandas as pd
        t = md_table(pd.DataFrame({"a": [1.5, float("nan")], "b": [True, False]}), "{:.1f}")
        self.assertEqual(t.splitlines()[0], "| a | b |")
        self.assertIn("| 1.5 | yes |", t)
        self.assertIn("|  | no |", t)

    def test_findings_rebuild_is_deterministic(self):
        r = self.res
        again = build_findings(r.benchmark, r.american, r.convergence, r.greeks, r.pricing_summary, r.hedge_base,
                               r.hedge_shock, r.conceptual)
        self.assertTrue(again.equals(r.findings))


if __name__ == "__main__":
    unittest.main()
