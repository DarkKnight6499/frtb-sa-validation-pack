"""Tests for the sensitivities-based method: formulas, correlations, scenarios, curvature."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frtb_val import columns as C  # noqa: E402
from frtb_val.frtb_sa import params as P  # noqa: E402
from frtb_val.frtb_sa import sbm  # noqa: E402

GIRR_ROW = {C.RISK_CLASS: C.GIRR, C.MEASURE: C.DELTA, C.CURVE: "YIELD"}


def girr(bucket, tenor, s):
    return {**GIRR_ROW, C.BUCKET: bucket, C.RISK_FACTOR: bucket, C.TENOR: tenor, C.SENSITIVITY: s, C.GAMMA: 0.0,
            C.INSTRUMENT: "x", C.DESK: "d"}


def frame(rows):
    return pd.DataFrame(rows)


class AggregationFormula(unittest.TestCase):
    def test_two_factor_hand_computed(self):
        # WS1 = 10, WS2 = -4, rho = 0.5: K = sqrt(100 + 16 + 2 * 0.5 * 10 * -4) = sqrt(76)
        ws = np.array([10.0, -4.0])
        rho = np.array([[1.0, 0.5], [0.5, 1.0]])
        self.assertAlmostEqual(sbm.bucket_capital(ws, rho), math.sqrt(76.0), places=12)

    def test_perfect_correlation_is_absolute_net_sum(self):
        ws = np.array([7.0, -3.0, 2.0])
        self.assertAlmostEqual(sbm.bucket_capital(ws, np.ones((3, 3))), abs(ws.sum()), places=10)

    def test_zero_correlation_is_root_sum_squares(self):
        ws = np.array([3.0, 4.0])
        self.assertAlmostEqual(sbm.bucket_capital(ws, np.eye(2)), 5.0, places=12)

    def test_floor_at_zero_under_the_root(self):
        ws = np.array([1.0, -1.0])
        rho = np.array([[1.0, 1.0], [1.0, 1.0]])
        self.assertEqual(sbm.bucket_capital(ws, rho), 0.0)

    def test_across_bucket_hand_computed(self):
        k = {"a": 3.0, "b": 4.0}
        s = {"a": 3.0, "b": 4.0}
        got = sbm.aggregate_buckets(k, s, lambda b, c: 0.5)
        self.assertAlmostEqual(got, math.sqrt(9 + 16 + 2 * 0.5 * 12), places=12)

    def test_alternative_specification_when_negative(self):
        k = {1: 0.1, 2: 0.1, 3: 0.1}
        s = {1: 1.0, 2: 1.0, 3: -2.0}
        got = sbm.aggregate_buckets(k, s, lambda b, c: 1.0)
        # S clipped to +-K = (0.1, 0.1, -0.1): 0.03 + 2 * (0.01 - 0.01 - 0.01) = 0.01
        self.assertAlmostEqual(got, 0.1, places=12)


class CorrelationScenarios(unittest.TestCase):
    def test_high_scenario_scaling_and_cap(self):
        self.assertAlmostEqual(float(sbm.scale_corr(0.5, "high")), 0.625)
        self.assertAlmostEqual(float(sbm.scale_corr(0.9, "high")), 1.0)

    def test_low_scenario_scaling(self):
        self.assertAlmostEqual(float(sbm.scale_corr(0.5, "low")), 0.375)   # max(0, 0.375)
        self.assertAlmostEqual(float(sbm.scale_corr(0.9, "low")), 0.8)     # max(0.8, 0.675)
        self.assertAlmostEqual(float(sbm.scale_corr(0.0, "low")), 0.0)

    def test_medium_unchanged(self):
        self.assertEqual(sbm.scale_corr(0.37, "medium"), 0.37)

    def test_same_sign_positions_bind_in_high_scenario(self):
        sens = frame([girr("USD", 1.0, 1e6), girr("USD", 5.0, 1e6)])
        res = sbm.sbm_capital(sens)
        self.assertEqual(res.selected, "high")
        self.assertAlmostEqual(res.charge, max(res.totals.values()))
        self.assertGreater(res.totals["high"], res.totals["medium"])
        self.assertGreater(res.totals["medium"], res.totals["low"])

    def test_offsetting_positions_bind_in_low_scenario(self):
        sens = frame([girr("USD", 1.0, 1e6), girr("USD", 5.0, -1e6)])
        res = sbm.sbm_capital(sens)
        self.assertEqual(res.selected, "low")
        self.assertEqual(res.charge, max(res.totals.values()))

    def test_total_is_max_of_scenario_totals_not_per_class_max(self):
        sens = frame([girr("USD", 1.0, 1e6), girr("USD", 5.0, 1e6),
                      {C.RISK_CLASS: C.FX, C.MEASURE: C.DELTA, C.BUCKET: "EUR", C.RISK_FACTOR: "EUR",
                       C.CURVE: "SPOT", C.TENOR: np.nan, C.SENSITIVITY: 1e6, C.GAMMA: 0.0, C.INSTRUMENT: "f",
                       C.DESK: "d"}])
        res = sbm.sbm_capital(sens)
        per_scenario = res.by_scenario[list(sbm.SCENARIOS)].sum()
        self.assertAlmostEqual(res.charge, per_scenario.max())


class Parameters(unittest.TestCase):
    def test_girr_tenor_formula_matches_published_table(self):
        # Spot values from the MAR21.46 Table 2 text.
        for (a, b, expected) in [(1, 5, 0.887), (0.25, 0.5, 0.970), (10, 30, 0.942), (0.25, 10, 0.400),
                                 (3, 5, 0.980), (2, 3, 0.985), (15, 20, 0.990), (0.5, 1, 0.970)]:
            self.assertAlmostEqual(P.girr_tenor_rho(a, b), expected, places=3, msg=(a, b))

    def test_girr_risk_weights(self):
        self.assertEqual(sbm.risk_weight(C.GIRR, C.DELTA, "USD", 0.25), 0.017)
        self.assertEqual(sbm.risk_weight(C.GIRR, C.DELTA, "USD", 2), 0.013)
        self.assertEqual(sbm.risk_weight(C.GIRR, C.DELTA, "USD", 30), 0.011)
        self.assertEqual(sbm.risk_weight(C.GIRR, C.DELTA, "USD", 1, "INFLATION"), 0.016)

    def test_csr_equity_fx_risk_weights(self):
        self.assertEqual(sbm.risk_weight(C.CSR, C.DELTA, 3, 5), 0.05)
        self.assertEqual(sbm.risk_weight(C.CSR, C.DELTA, 17, 5), 0.015)
        self.assertEqual(sbm.risk_weight(C.EQ, C.DELTA, 8, np.nan, "SPOT"), 0.50)
        self.assertAlmostEqual(sbm.risk_weight(C.EQ, C.DELTA, 8, np.nan, "REPO"), 0.005)
        self.assertEqual(sbm.risk_weight(C.FX, C.DELTA, "EUR", np.nan, "SPOT"), 0.15)

    def test_vega_risk_weights(self):
        self.assertAlmostEqual(sbm.risk_weight(C.EQ, C.VEGA, 8, 1.0), 0.55 * math.sqrt(2), places=12)  # 77.78%
        self.assertEqual(sbm.risk_weight(C.EQ, C.VEGA, 10, 1.0), 1.0)      # small cap, LH 60: capped at 100%
        self.assertEqual(sbm.risk_weight(C.FX, C.VEGA, "EUR", 1.0), 1.0)
        self.assertEqual(sbm.risk_weight(C.GIRR, C.VEGA, "USD", 1.0), 1.0)

    def test_csr_gamma_values(self):
        self.assertEqual(P.csr_gamma(1, 2), 0.75)
        self.assertEqual(P.csr_gamma(1, 9), 0.50)       # same sector, IG against HY
        self.assertEqual(P.csr_gamma(3, 17), 0.45)
        self.assertEqual(P.csr_gamma(16, 4), 0.0)
        self.assertEqual(P.csr_gamma(17, 18), 0.75)
        self.assertEqual(P.csr_gamma(1, 10), 0.5 * 0.75)
        self.assertEqual(P.csr_gamma(4, 4), 1.0)

    def test_csr_gamma_is_symmetric(self):
        for b in range(1, 19):
            for c in range(1, 19):
                self.assertEqual(P.csr_gamma(b, c), P.csr_gamma(c, b))

    def test_equity_gamma_values(self):
        self.assertEqual(P.eq_gamma(1, 7), 0.15)
        self.assertEqual(P.eq_gamma(11, 3), 0.0)
        self.assertEqual(P.eq_gamma(12, 13), 0.75)
        self.assertEqual(P.eq_gamma(5, 12), 0.45)

    def test_girr_cross_currency_gamma(self):
        self.assertEqual(sbm.inter_bucket_gamma(C.GIRR, "USD", "EUR"), 0.5)
        self.assertEqual(sbm.inter_bucket_gamma(C.FX, "EUR", "GBP"), 0.6)

    def test_equity_within_bucket_rho(self):
        f = pd.DataFrame([{C.RISK_FACTOR: "A", C.CURVE: "SPOT", C.TENOR: np.nan},
                          {C.RISK_FACTOR: "B", C.CURVE: "SPOT", C.TENOR: np.nan},
                          {C.RISK_FACTOR: "A", C.CURVE: "REPO", C.TENOR: np.nan},
                          {C.RISK_FACTOR: "B", C.CURVE: "REPO", C.TENOR: np.nan}])
        r = sbm.intra_bucket_rho(C.EQ, C.DELTA, 8, f)
        self.assertAlmostEqual(r[0, 1], 0.25)
        self.assertAlmostEqual(r[0, 2], 0.999)
        self.assertAlmostEqual(r[0, 3], 0.25 * 0.999)

    def test_csr_within_bucket_rho_product_form(self):
        f = pd.DataFrame([{C.RISK_FACTOR: "APPLE", C.CURVE: "BOND", C.TENOR: 5.0},
                          {C.RISK_FACTOR: "GOOGLE", C.CURVE: "CDS", C.TENOR: 10.0}])
        r = sbm.intra_bucket_rho(C.CSR, C.DELTA, 6, f)
        self.assertAlmostEqual(r[0, 1], 0.35 * 0.65 * 0.999, places=12)   # 22.73%, MAR21.54 footnote 18

    def test_girr_xccy_and_inflation_correlation(self):
        f = pd.DataFrame([{C.CURVE: "YIELD", C.TENOR: 5.0, C.RISK_FACTOR: "x"},
                          {C.CURVE: "INFLATION", C.TENOR: 5.0, C.RISK_FACTOR: "x"},
                          {C.CURVE: "XCCY_USD", C.TENOR: 5.0, C.RISK_FACTOR: "x"}])
        r = sbm.intra_bucket_rho(C.GIRR, C.DELTA, "EUR", f)
        self.assertEqual(r[0, 1], 0.40)
        self.assertEqual(r[0, 2], 0.0)

    def test_vega_maturity_correlation(self):
        self.assertAlmostEqual(P.vega_maturity_rho(1, 5), math.exp(-0.01 * 4), places=12)


class DeltaCharge(unittest.TestCase):
    def test_single_factor_charge_is_risk_weighted_sensitivity(self):
        res = sbm.sbm_capital(frame([girr("USD", 5.0, 2e6)]))
        self.assertAlmostEqual(res.totals["medium"], 0.011 * 2e6, places=6)

    def test_charge_decreases_with_hedge_offset(self):
        base = frame([girr("USD", 10.0, 5e6)])
        hedged = frame([girr("USD", 10.0, 5e6), girr("USD", 10.0, -2e6)])
        self.assertLess(sbm.sbm_capital(hedged).charge, sbm.sbm_capital(base).charge)

    def test_full_same_factor_hedge_gives_zero(self):
        sens = frame([girr("USD", 10.0, 5e6), girr("USD", 10.0, -5e6)])
        self.assertEqual(sbm.sbm_capital(sens).charge, 0.0)

    def test_cross_tenor_hedge_leaves_residual_basis(self):
        sens = frame([girr("USD", 5.0, 5e6), girr("USD", 10.0, -5e6)])
        residual = sbm.sbm_capital(sens).totals["medium"]
        self.assertGreater(residual, 0.0)
        self.assertLess(residual, 0.011 * 5e6 * 2)

    def test_cross_currency_diversification(self):
        usd = frame([girr("USD", 5.0, 1e6)])
        both = frame([girr("USD", 5.0, 1e6), girr("EUR", 5.0, 1e6)])
        single = sbm.sbm_capital(usd).totals["medium"]
        total = sbm.sbm_capital(both).totals["medium"]
        self.assertLess(total, 2 * single)
        self.assertAlmostEqual(total, math.sqrt(2 * single ** 2 + 2 * 0.5 * single ** 2), places=6)

    def test_equity_other_bucket_is_simple_sum_of_absolutes(self):
        rows = [{C.RISK_CLASS: C.EQ, C.MEASURE: C.DELTA, C.BUCKET: 11, C.RISK_FACTOR: n, C.CURVE: "SPOT",
                 C.TENOR: np.nan, C.SENSITIVITY: s, C.GAMMA: 0.0, C.INSTRUMENT: "i", C.DESK: "d"}
                for n, s in (("A", 1e6), ("B", -1e6))]
        cap, _ = sbm.delta_vega_capital(sbm.net_sensitivities(frame(rows)), C.EQ, C.DELTA, "medium")
        self.assertAlmostEqual(cap, 0.70 * 2e6, places=6)

    def test_netting_same_risk_factor_across_instruments(self):
        sens = frame([{**girr("USD", 5.0, 1e6), C.INSTRUMENT: "a"}, {**girr("USD", 5.0, -1e6), C.INSTRUMENT: "b"}])
        net = sbm.net_sensitivities(sens)
        self.assertEqual(len(net), 1)
        self.assertEqual(float(net[C.SENSITIVITY].iloc[0]), 0.0)

    def test_empty_inputs_give_zero(self):
        empty = pd.DataFrame(columns=[C.RISK_CLASS, C.MEASURE, C.BUCKET, C.RISK_FACTOR, C.CURVE, C.TENOR,
                                      C.SENSITIVITY, C.GAMMA])
        self.assertEqual(sbm.sbm_capital(empty).charge, 0.0)


class Curvature(unittest.TestCase):
    def test_k_with_psi_rule_both_negative_gives_zero(self):
        up, dn = np.array([-5.0]), np.array([-3.0])
        k, _ = sbm.curvature_bucket(up, dn, np.eye(1))
        self.assertEqual(k, 0.0)

    def test_two_factor_hand_computed(self):
        up, dn = np.array([10.0, 6.0]), np.array([1.0, 1.0])
        rho = np.array([[1.0, 0.0625], [0.0625, 1.0]])
        k, s = sbm.curvature_bucket(up, dn, rho)
        self.assertAlmostEqual(k, math.sqrt(100 + 36 + 2 * 0.0625 * 60), places=12)
        self.assertEqual(s, 16.0)

    def test_psi_removes_cross_term_between_negative_and_negative_only(self):
        up = np.array([10.0, -6.0])
        rho = np.array([[1.0, 0.5], [0.5, 1.0]])
        k, _ = sbm.curvature_bucket(up, np.zeros(2), rho)
        self.assertAlmostEqual(k, math.sqrt(100 + 0 + 2 * 0.5 * 10 * -6), places=12)   # psi = 1 for mixed signs

    def test_across_bucket_uses_squared_gamma(self):
        curv = pd.DataFrame([{C.RISK_CLASS: C.GIRR, C.BUCKET: "USD", C.RISK_FACTOR: "USD", C.CVR_UP: 10.0, C.CVR_DOWN: 0.0},
                             {C.RISK_CLASS: C.GIRR, C.BUCKET: "EUR", C.RISK_FACTOR: "EUR", C.CVR_UP: 10.0, C.CVR_DOWN: 0.0}])
        cap, _ = sbm.curvature_capital(curv, C.GIRR, "medium")
        self.assertAlmostEqual(cap, math.sqrt(100 + 100 + 2 * 0.25 * 100), places=12)   # 0.5 squared = 0.25

    def test_curvature_scenario_scaling_applies_to_squared_correlation(self):
        curv = pd.DataFrame([{C.RISK_CLASS: C.GIRR, C.BUCKET: "USD", C.RISK_FACTOR: "USD", C.CVR_UP: 10.0, C.CVR_DOWN: 0.0},
                             {C.RISK_CLASS: C.GIRR, C.BUCKET: "EUR", C.RISK_FACTOR: "EUR", C.CVR_UP: 10.0, C.CVR_DOWN: 0.0}])
        hi, _ = sbm.curvature_capital(curv, C.GIRR, "high")
        self.assertAlmostEqual(hi, math.sqrt(200 + 2 * 0.3125 * 100), places=12)   # 1.25 * 0.25

    def test_selects_larger_of_up_and_down(self):
        up, dn = np.array([4.0]), np.array([9.0])
        k, s = sbm.curvature_bucket(up, dn, np.eye(1))
        self.assertEqual((k, s), (9.0, 9.0))


if __name__ == "__main__":
    unittest.main()
