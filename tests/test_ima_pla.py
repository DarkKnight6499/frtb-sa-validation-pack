"""Tests for expected shortfall, liquidity horizons, stress selection, RFET, backtesting and the PLA test."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frtb_val import columns as C  # noqa: E402
from frtb_val.ima import backtest as B  # noqa: E402
from frtb_val.ima import es as E  # noqa: E402
from frtb_val.ima import factors as F  # noqa: E402
from frtb_val.ima import nmrf as N  # noqa: E402
from frtb_val.ima.pipeline import build_inputs  # noqa: E402
from frtb_val.mathutil import (binom_cdf, expected_shortfall, ks_pvalue, ks_statistic, norm_pdf, norm_ppf,  # noqa: E402
                               rankdata, spearman, value_at_risk)
from frtb_val.pla import pla_test as PLA  # noqa: E402


def normal_sample(n=20000):
    return np.array([norm_ppf((i + 0.5) / n) for i in range(n)])


class ExpectedShortfall(unittest.TestCase):
    def test_matches_normal_analytic(self):
        x = normal_sample()
        expected = norm_pdf(norm_ppf(0.975)) / 0.025
        self.assertAlmostEqual(expected_shortfall(x, 0.975), expected, delta=0.01 * expected)

    def test_monotone_in_confidence_level(self):
        x = np.random.default_rng(3).standard_t(4, 5000)
        levels = [0.90, 0.95, 0.975, 0.99, 0.995]
        values = [expected_shortfall(x, a) for a in levels]
        self.assertTrue(all(b > a for a, b in zip(values, values[1:])))

    def test_es_at_least_var(self):
        x = np.random.default_rng(4).normal(size=2000)
        self.assertGreaterEqual(expected_shortfall(x, 0.975), value_at_risk(x, 0.975))

    def test_es_of_sample_with_fractional_tail(self):
        pnl = -np.arange(1, 41, dtype=float)       # losses 1..40, 2.5% of 40 = one observation
        self.assertAlmostEqual(expected_shortfall(pnl, 0.975), 40.0)
        pnl2 = -np.arange(1, 21, dtype=float)       # 0.5 of an observation: still the worst loss
        self.assertAlmostEqual(expected_shortfall(pnl2, 0.975), 20.0)

    def test_scale_and_translation_behaviour(self):
        x = np.random.default_rng(5).normal(size=1000)
        self.assertAlmostEqual(expected_shortfall(2 * x, 0.975), 2 * expected_shortfall(x, 0.975))
        self.assertAlmostEqual(expected_shortfall(x + 1, 0.975), expected_shortfall(x, 0.975) - 1)

    def test_subadditive(self):
        rng = np.random.default_rng(6)
        a, b = rng.standard_t(3, 3000), rng.standard_t(3, 3000)
        self.assertLessEqual(expected_shortfall(a + b), expected_shortfall(a) + expected_shortfall(b) + 1e-9)

    def test_regulatory_constants(self):
        self.assertEqual(E.LIQUIDITY_HORIZONS, (10, 20, 40, 60, 120))
        self.assertEqual(E.ALPHA, 0.975)
        self.assertEqual(E.IMCC_RHO, 0.5)


class LiquidityHorizonCascade(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(10)
        self.contrib = self.rng.standard_t(5, (250, 6)) * np.array([1, 2, 3, 1, 2, 1])

    def test_reduces_to_plain_es_when_all_horizons_are_ten(self):
        got = E.cascade_es(self.contrib, np.full(6, 10))
        self.assertAlmostEqual(got, expected_shortfall(self.contrib.sum(axis=1), 0.975), places=12)

    def test_two_horizon_hand_computation(self):
        lh = np.array([10, 20, 10, 10, 10, 10])
        want = math.sqrt(expected_shortfall(self.contrib.sum(axis=1)) ** 2
                         + (expected_shortfall(self.contrib[:, [1]].sum(axis=1)) * math.sqrt((20 - 10) / 10)) ** 2)
        self.assertAlmostEqual(E.cascade_es(self.contrib, lh), want, places=12)

    def test_longer_horizons_increase_es(self):
        base = E.cascade_es(self.contrib, np.full(6, 10))
        longer = E.cascade_es(self.contrib, np.array([10, 40, 60, 10, 120, 20]))
        self.assertGreater(longer, base)

    def test_each_horizon_step_is_monotone(self):
        values = [E.cascade_es(self.contrib, np.array([10, h, 10, 10, 10, 10])) for h in (10, 20, 40, 60, 120)]
        self.assertTrue(all(b >= a for a, b in zip(values, values[1:])))

    def test_all_120_day_factors_scale_es_like_the_cascade_formula(self):
        # With every factor at 120 days the nested sets coincide, so ES = ES_T * sqrt(1 + 1 + 2 + 2 + 6)
        es_t = expected_shortfall(self.contrib.sum(axis=1))
        got = E.cascade_es(self.contrib, np.full(6, 120))
        self.assertAlmostEqual(got, es_t * math.sqrt(1 + 1 + 2 + 2 + 6), places=10)

    def test_liquidity_horizon_mapping_for_synthetic_factor_classes(self):
        self.assertEqual(F.liquidity_horizon(C.GIRR, C.DELTA, "USD"), 10)
        self.assertEqual(F.liquidity_horizon(C.GIRR, C.VEGA, "USD"), 60)
        self.assertEqual(F.liquidity_horizon(C.CSR, C.DELTA, 3), 40)
        self.assertEqual(F.liquidity_horizon(C.CSR, C.DELTA, 11), 60)
        self.assertEqual(F.liquidity_horizon(C.CSR, C.VEGA, 3), 120)
        self.assertEqual(F.liquidity_horizon(C.EQ, C.DELTA, 8), 10)
        self.assertEqual(F.liquidity_horizon(C.EQ, C.VEGA, 10), 60)
        self.assertEqual(F.liquidity_horizon(C.FX, C.VEGA, "EUR"), 40)


class StressedEs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inp = build_inputs()
        c10, lh, broad = cls.inp.contrib10, cls.inp.lh, cls.inp.broad
        cls.reduced = E.select_reduced_set(c10[-250:], lh, broad)
        cls.res = E.stressed_es(c10, lh, cls.reduced, broad)

    def test_stress_window_recovers_the_planted_regime(self):
        start = self.res.stress_start
        overlap = len(set(range(start, start + 250 + 9)) & set(range(700, 950)))
        self.assertGreaterEqual(overlap, 200)

    def test_stressed_es_exceeds_current_es(self):
        self.assertGreater(self.res.es_stressed, 2 * self.res.es_fc)

    def test_reduced_set_explains_at_least_75_percent(self):
        self.assertGreaterEqual(self.res.explained, 0.75)

    def test_ratio_is_floored_at_one(self):
        self.assertGreaterEqual(self.res.ratio_applied, 1.0)
        self.assertAlmostEqual(self.res.es_stressed, self.res.es_rs * self.res.ratio_applied)

    def test_ratio_floor_binds_when_reduced_set_overstates_risk(self):
        c10, lh, broad = self.inp.contrib10, self.inp.lh, self.inp.broad
        # A reduced set holding only the most unhedged factor can exceed the full-set ES, and the ratio must floor at 1.
        standalone = [expected_shortfall(c10[-250:, i]) for i in range(c10.shape[1])]
        r = E.stressed_es(c10, lh, [int(np.argmax(standalone))], broad)
        if r.ratio_raw < 1.0:
            self.assertEqual(r.ratio_applied, 1.0)
        self.assertGreaterEqual(r.ratio_applied, 1.0)

    def test_imcc_is_blend_of_unconstrained_and_constrained(self):
        r = self.res
        self.assertAlmostEqual(r.imcc, 0.5 * r.imcc_unconstrained + 0.5 * r.imcc_constrained_sum)

    def test_constrained_sum_is_at_least_unconstrained(self):
        # Class-level ES values are summed without diversification across classes.
        self.assertGreaterEqual(self.res.imcc_constrained_sum, self.res.imcc_unconstrained * 0.999)

    def test_panel_is_reproducible_and_stress_has_higher_variance(self):
        again = build_inputs()
        np.testing.assert_array_equal(again.panel.to_numpy(), self.inp.panel.to_numpy())
        x = self.inp.panel.to_numpy()
        self.assertGreater(x[700:950].std(), 1.8 * x[:600].std())


class RiskFactorEligibility(unittest.TestCase):
    def test_evenly_spread_24_observations_pass_criterion_1(self):
        res = N.rfet(np.arange(0, 365, 15)[:24])
        self.assertEqual(res["n_obs"], 24)
        self.assertTrue(res["criterion_1"])
        self.assertTrue(res["modellable"])

    def test_23_observations_fail(self):
        res = N.rfet(np.linspace(0, 364, 23).astype(int))
        self.assertFalse(res["modellable"])

    def test_clustered_24_observations_fail_the_90_day_rule(self):
        res = N.rfet(np.arange(24) * 2)      # all within the first 48 days
        self.assertEqual(res["n_obs"], 24)
        self.assertFalse(res["criterion_1"])
        self.assertFalse(res["modellable"])

    def test_100_observations_pass_criterion_2_even_if_clustered(self):
        res = N.rfet(np.arange(100))
        self.assertFalse(res["criterion_1"])
        self.assertTrue(res["criterion_2"])
        self.assertTrue(res["modellable"])

    def test_99_clustered_observations_fail(self):
        self.assertFalse(N.rfet(np.arange(99))["modellable"])

    def test_one_observation_per_day_counts(self):
        res = N.rfet([5] * 200)
        self.assertEqual(res["n_obs"], 1)
        self.assertFalse(res["modellable"])

    def test_window_with_exactly_four_observations_is_acceptable(self):
        days = np.concatenate([np.arange(0, 365, 90)[:5], np.arange(1, 365, 15)])
        res = N.rfet(days)
        self.assertGreaterEqual(res["min_obs_in_90d"], 4)


class StressScenarioCharge(unittest.TestCase):
    def test_uses_at_least_20_day_horizon(self):
        changes = np.full(250, 0.01)
        # LH 10 is floored to 20, so a short position loses 20 * 0.01 per scenario
        self.assertAlmostEqual(N.stress_scenario_charge(changes, -1.0, 0.0, 10), 0.20, places=12)
        self.assertAlmostEqual(N.stress_scenario_charge(changes, -1.0, 0.0, 60), 0.60, places=12)

    def test_profit_scenario_gives_zero_charge(self):
        self.assertEqual(N.stress_scenario_charge(np.full(250, 0.01), 1.0, 0.0, 20), 0.0)

    def test_ses_single_factor_equals_itself(self):
        self.assertAlmostEqual(N.ses(np.array([5.0])), 5.0)

    def test_ses_two_factors_hand_computed(self):
        self.assertAlmostEqual(N.ses(np.array([1.0, 1.0])), math.sqrt((0.6 * 2) ** 2 + 2 * (1 - 0.36)), places=12)

    def test_ses_idiosyncratic_sets_add_in_quadrature_then_sum(self):
        got = N.ses(np.zeros(0), np.array([3.0, 4.0]), np.array([6.0, 8.0]))
        self.assertAlmostEqual(got, 5.0 + 10.0)


class Backtesting(unittest.TestCase):
    def test_binomial_cumulative_probabilities_match_basel_table(self):
        table = {0: 0.0811, 1: 0.2858, 2: 0.5432, 3: 0.7581, 4: 0.8922, 5: 0.9588, 6: 0.9863, 7: 0.9960, 8: 0.9989,
                 9: 0.9997}
        for k, v in table.items():
            self.assertAlmostEqual(binom_cdf(k, 250, 0.01), v, places=4)

    def test_zone_boundaries_for_250_days(self):
        self.assertEqual(B.zone_boundaries(250, 0.99), (5, 10))

    def test_traffic_light_zones(self):
        zones = [B.traffic_light(k) for k in range(0, 13)]
        self.assertEqual(zones[:5], ["green"] * 5)
        self.assertEqual(zones[5:10], ["amber"] * 5)
        self.assertEqual(zones[10:], ["red"] * 3)

    def test_multipliers(self):
        self.assertEqual([B.multiplier(k) for k in (0, 4, 5, 6, 7, 8, 9, 10, 15)],
                         [1.5, 1.5, 1.7, 1.76, 1.83, 1.88, 1.92, 2.0, 2.0])

    def test_zone_boundaries_move_with_sample_size(self):
        amber500, red500 = B.zone_boundaries(500, 0.99)
        self.assertGreater(amber500, 5)
        self.assertGreater(red500, 10)

    def test_desk_failure_thresholds(self):
        self.assertFalse(B.desk_backtest_fail(12, 30))
        self.assertTrue(B.desk_backtest_fail(13, 0))
        self.assertTrue(B.desk_backtest_fail(0, 31))

    def test_binomial_exceedance_test(self):
        t = B.binomial_exceedance_test(10, 250, 0.99)
        self.assertAlmostEqual(t["p_too_many"], 1 - binom_cdf(9, 250, 0.01))
        self.assertAlmostEqual(t["expected"], 2.5)
        self.assertLess(t["p_too_many"], 0.001)

    def test_p_value_decreases_with_more_exceptions(self):
        ps = [B.binomial_exceedance_test(k, 250, 0.99)["p_too_many"] for k in range(0, 15)]
        self.assertTrue(all(b < a for a, b in zip(ps, ps[1:])))

    def test_kupiec_zero_lr_at_expected_rate(self):
        r = B.kupiec_pof(5, 500, 0.99)
        self.assertAlmostEqual(r["lr"], 0.0, places=10)
        self.assertAlmostEqual(r["p_value"], 1.0, places=6)

    def test_kupiec_rejects_many_exceptions(self):
        self.assertLess(B.kupiec_pof(15, 250, 0.99)["p_value"], 0.001)

    def test_exceptions_counted_on_constructed_series(self):
        rng = np.random.default_rng(1)
        pnl = rng.normal(0, 1, 800)
        pnl[-250:] = rng.normal(0, 1, 250)
        res = B.backtest(pnl, 0.99)
        var = res.var_series
        manual = int(np.sum(-pnl[-250:] > var))
        self.assertEqual(res.exceptions, manual)

    def test_understated_var_gives_more_exceptions(self):
        pnl = np.random.default_rng(2).normal(0, 1, 900)
        full = B.backtest(pnl, 0.99).exceptions
        weak = B.backtest(pnl, 0.99, scale=0.6).exceptions
        self.assertGreater(weak, full)

    def test_var_matches_empirical_quantile(self):
        x = np.arange(1, 101, dtype=float)
        self.assertEqual(value_at_risk(-x, 0.99), 99.0)


class MathUtilities(unittest.TestCase):
    def test_spearman_known_value(self):
        x = [106, 86, 100, 101, 99, 103, 97, 113, 112, 110]
        y = [7, 0, 27, 50, 28, 29, 20, 12, 6, 17]
        self.assertAlmostEqual(spearman(x, y), -29 / 165, places=12)

    def test_spearman_extremes_and_ties(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertEqual(rankdata([10, 20, 20, 30]).tolist(), [1.0, 2.5, 2.5, 4.0])

    def test_spearman_invariant_to_monotone_transform(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=100), rng.normal(size=100)
        self.assertAlmostEqual(spearman(a, b), spearman(np.exp(a), b ** 3))

    def test_ks_known_values(self):
        self.assertAlmostEqual(ks_statistic([1, 2, 3], [2, 3, 4]), 1 / 3)
        self.assertEqual(ks_statistic([1, 2, 3], [1, 2, 3]), 0.0)
        self.assertEqual(ks_statistic([1, 2, 3], [10, 11, 12]), 1.0)

    def test_ks_is_symmetric(self):
        rng = np.random.default_rng(2)
        a, b = rng.normal(size=80), rng.normal(0.5, 1, 90)
        self.assertAlmostEqual(ks_statistic(a, b), ks_statistic(b, a))

    def test_ks_ecdf_uses_0_004_steps_for_250_observations(self):
        a = np.arange(250.0)
        b = a + 0.5
        self.assertAlmostEqual(ks_statistic(a, b), 0.004)

    def test_ks_p_values_match_basel_thresholds(self):
        self.assertAlmostEqual(ks_pvalue(0.09, 250, 250), 0.264, delta=0.002)
        self.assertAlmostEqual(ks_pvalue(0.12, 250, 250), 0.055, delta=0.001)

    def test_ks_p_value_monotone(self):
        ps = [ks_pvalue(d, 250, 250) for d in (0.05, 0.09, 0.12, 0.2)]
        self.assertTrue(all(b < a for a, b in zip(ps, ps[1:])))

    def test_normal_ppf_round_trip(self):
        for p in (0.001, 0.025, 0.5, 0.975, 0.999):
            from frtb_val.mathutil import norm_cdf
            self.assertAlmostEqual(norm_cdf(norm_ppf(p)), p, places=12)
        self.assertAlmostEqual(norm_ppf(0.975), 1.959963984540054, places=9)


class PLATest(unittest.TestCase):
    def test_zone_rules(self):
        self.assertEqual(PLA.pla_zone(0.85, 0.05), "green")
        self.assertEqual(PLA.pla_zone(0.75, 0.05), "amber")
        self.assertEqual(PLA.pla_zone(0.85, 0.10), "amber")
        self.assertEqual(PLA.pla_zone(0.65, 0.05), "red")
        self.assertEqual(PLA.pla_zone(0.85, 0.13), "red")

    def test_zone_boundaries_are_not_green(self):
        self.assertEqual(PLA.pla_zone(0.80, 0.05), "amber")     # needs corr above 0.80
        self.assertEqual(PLA.pla_zone(0.85, 0.09), "amber")     # needs KS below 0.09
        self.assertEqual(PLA.pla_zone(0.70, 0.05), "amber")     # red needs corr below 0.70
        self.assertEqual(PLA.pla_zone(0.85, 0.12), "amber")     # red needs KS above 0.12

    def test_identical_series_are_green(self):
        x = np.random.default_rng(3).normal(size=250)
        r = PLA.pla_test(x, x)
        self.assertEqual((r.spearman, r.ks, r.zone), (1.0, 0.0, "green"))

    def test_independent_series_are_red(self):
        rng = np.random.default_rng(4)
        r = PLA.pla_test(rng.normal(size=250), rng.normal(size=250))
        self.assertEqual(r.zone, "red")
        self.assertLess(r.spearman, 0.3)

    def test_noisy_replication_lands_between(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=250)
        r = PLA.pla_test(x, x + rng.normal(0, 0.7, 250))
        self.assertIn(r.zone, ("amber", "green"))
        self.assertLess(r.spearman, 0.95)

    def test_uses_most_recent_250_observations_only(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=400)
        junk = x.copy()
        junk[:150] = rng.normal(size=150)
        self.assertEqual(PLA.pla_test(x, junk).spearman, 1.0)

    def test_scale_mismatch_is_caught_by_ks_not_spearman(self):
        x = np.random.default_rng(7).normal(size=250)
        r = PLA.pla_test(x, 3 * x)
        self.assertEqual(r.spearman, 1.0)
        self.assertGreater(r.ks, 0.12)
        self.assertEqual(r.zone, "red")

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            PLA.pla_test(np.zeros(250), np.zeros(240))

    def test_desk_eligibility_mapping(self):
        self.assertEqual(PLA.desk_eligibility("green", 3, 10), "IMA eligible")
        self.assertIn("surcharge", PLA.desk_eligibility("amber", 3, 10))
        self.assertIn("PLA red", PLA.desk_eligibility("red", 0, 0))
        self.assertIn("backtesting", PLA.desk_eligibility("green", 13, 0))


if __name__ == "__main__":
    unittest.main()
