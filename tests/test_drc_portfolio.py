"""Tests for the default risk charge, RRAO, instruments, sensitivities and standardised approach totals."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frtb_val import columns as C  # noqa: E402
from frtb_val.frtb_sa.capital import sa_capital  # noqa: E402
from frtb_val.frtb_sa.drc import drc_capital, gross_jtd, net_jtd, rrao_capital  # noqa: E402
from frtb_val.instruments import CDS, IRS, Bond, EqOption, Exotic, FXForward, FXOption, Stock, cds_legs  # noqa: E402
from frtb_val.market import build_market  # noqa: E402
from frtb_val.portfolio import (book_value, build_book, curvature, drc_positions, rrao_positions,  # noqa: E402
                                sensitivities)
from frtb_val.pricing import bs_greeks  # noqa: E402


def pos(obligor, notional, pnl=0.0, rating="BBB", seniority="senior", bucket="corporate", maturity=5.0):
    return {C.OBLIGOR: obligor, C.SENIORITY: seniority, C.RATING: rating, C.DRC_BUCKET: bucket,
            C.NOTIONAL: notional, C.PNL_TO_DATE: pnl, C.MATURITY: maturity, "kind": "t"}


class DefaultRiskCharge(unittest.TestCase):
    def test_gross_jtd_long_uses_lgd_notional_plus_pnl(self):
        g = gross_jtd(pd.DataFrame([pos("A", 100.0, pnl=-2.0)]))
        self.assertAlmostEqual(g["gross_jtd"].iloc[0], 0.75 * 100 - 2.0)

    def test_gross_jtd_long_floored_at_zero_and_short_capped_at_zero(self):
        g = gross_jtd(pd.DataFrame([pos("A", 10.0, pnl=-50.0), pos("B", -10.0, pnl=50.0)]))
        self.assertEqual(g["gross_jtd"].tolist(), [0.0, 0.0])

    def test_gross_jtd_short(self):
        g = gross_jtd(pd.DataFrame([pos("A", -100.0, pnl=2.0)]))
        self.assertAlmostEqual(g["gross_jtd"].iloc[0], -(75.0 - 2.0))

    def test_equity_lgd_is_100_percent(self):
        g = gross_jtd(pd.DataFrame([pos("A", 100.0, seniority="equity", rating="UNRATED")]))
        self.assertAlmostEqual(g["gross_jtd"].iloc[0], 100.0)

    def test_maturity_weighting_and_floor(self):
        g = gross_jtd(pd.DataFrame([pos("A", 100.0, maturity=0.5), pos("B", 100.0, maturity=0.1),
                                    pos("C", 100.0, maturity=3.0)]))
        self.assertEqual(g["scaled_jtd"].round(6).tolist(), [37.5, 18.75, 75.0])

    def test_hedge_benefit_ratio_hand_computed(self):
        res = drc_capital(pd.DataFrame([pos("A", 100.0), pos("B", -50.0)]))
        # long JTD 75, short JTD 37.5, HBR = 75 / 112.5, RW(BBB) = 6%: 4.5 - (2/3) * 2.25 = 3.0
        self.assertAlmostEqual(res.by_bucket["hbr"].iloc[0], 75.0 / 112.5)
        self.assertAlmostEqual(res.total, 3.0, places=10)

    def test_long_only_charge_is_risk_weighted_jtd(self):
        res = drc_capital(pd.DataFrame([pos("A", 100.0, rating="BB")]))
        self.assertAlmostEqual(res.total, 0.15 * 75.0)

    def test_short_only_charge_is_zero(self):
        self.assertEqual(drc_capital(pd.DataFrame([pos("A", -100.0)])).total, 0.0)

    def test_adding_short_hedge_reduces_charge(self):
        base = drc_capital(pd.DataFrame([pos("A", 100.0)])).total
        hedged = drc_capital(pd.DataFrame([pos("A", 100.0), pos("B", -40.0)])).total
        self.assertLess(hedged, base)

    def test_same_obligor_offsets_in_full(self):
        res = drc_capital(pd.DataFrame([pos("A", 100.0), pos("A", -100.0)]))
        self.assertEqual(res.total, 0.0)

    def test_short_equity_offsets_long_bond_but_not_reverse(self):
        long_bond_short_eq = net_jtd(gross_jtd(pd.DataFrame([
            pos("A", 100.0), pos("A", -20.0, seniority="equity", rating="UNRATED")])))
        self.assertAlmostEqual(long_bond_short_eq["net_long"].iloc[0], 75.0 - 20.0)
        long_eq_short_bond = net_jtd(gross_jtd(pd.DataFrame([
            pos("A", 20.0, seniority="equity", rating="UNRATED"), pos("A", -100.0)])))
        self.assertAlmostEqual(long_eq_short_bond["net_long"].iloc[0], 20.0)    # unchanged
        self.assertAlmostEqual(long_eq_short_bond["net_short"].iloc[0], 75.0)

    def test_buckets_do_not_diversify(self):
        a = drc_capital(pd.DataFrame([pos("A", 100.0)])).total
        b = drc_capital(pd.DataFrame([pos("S", 100.0, bucket="sovereign", rating="AA")])).total
        both = drc_capital(pd.DataFrame([pos("A", 100.0), pos("S", 100.0, bucket="sovereign", rating="AA")])).total
        self.assertAlmostEqual(both, a + b)

    def test_empty_positions(self):
        self.assertEqual(drc_capital(pd.DataFrame(columns=[C.OBLIGOR])).total, 0.0)


class ResidualRiskAddOn(unittest.TestCase):
    def test_risk_weights(self):
        pos_df = pd.DataFrame([{C.INSTRUMENT: "a", "rrao_type": "exotic", C.NOTIONAL: 100.0},
                               {C.INSTRUMENT: "b", "rrao_type": "other", C.NOTIONAL: 1000.0}])
        self.assertAlmostEqual(rrao_capital(pos_df), 1.0 + 1.0)

    def test_empty(self):
        self.assertEqual(rrao_capital(pd.DataFrame(columns=[C.INSTRUMENT, "rrao_type", C.NOTIONAL])), 0.0)


class Instruments(unittest.TestCase):
    def setUp(self):
        self.m = build_market()

    def test_irs_at_par_rate_has_zero_value(self):
        m = self.m
        annuity = sum(m.df("USD", t) for t in range(1, 6))
        par = (1 - m.df("USD", 5)) / annuity
        self.assertAlmostEqual(IRS("i", "R", notional=1e6, fixed=par, maturity=5).value(m), 0.0, places=6)

    def test_payer_and_receiver_are_opposite(self):
        a = IRS("i", "R", ccy="EUR", notional=1e6, fixed=0.03, maturity=4, payer=True).value(self.m)
        b = IRS("j", "R", ccy="EUR", notional=1e6, fixed=0.03, maturity=4, payer=False).value(self.m)
        self.assertAlmostEqual(a, -b)

    def test_risk_free_bond_at_par_yield_prices_at_face(self):
        m = self.m
        annuity = sum(m.df("USD", t) for t in range(1, 4))
        par = (1 - m.df("USD", 3)) / annuity
        self.assertAlmostEqual(Bond("b", "R", face=100.0, coupon=par, maturity=3).value(m), 100.0, places=8)

    def test_credit_spread_lowers_bond_value(self):
        m = self.m
        rf = Bond("b", "R", face=100.0, coupon=0.05, maturity=5).value(m)
        cr = Bond("b", "R", face=100.0, coupon=0.05, maturity=5, issuer="FIN_A").value(m)
        self.assertLess(cr, rf)

    def test_cds_at_par_coupon_has_zero_value(self):
        m = self.m
        prot, prem = cds_legs(m, "IND_A", 5, 1.0)       # premium per unit coupon
        par = prot / prem
        self.assertAlmostEqual(CDS("c", "C", name="IND_A", notional=1e6, coupon=par, maturity=5).value(m), 0.0, places=6)

    def test_protection_buyer_gains_when_spread_widens(self):
        m = self.m
        c = CDS("c", "C", name="IND_A", notional=1e6, coupon=0.01, maturity=5, buyer=True)
        wider = m.bumped("csr", "IND_A", 3, 0.005)
        self.assertGreater(c.value(wider), c.value(m))

    def test_fx_forward_at_forward_rate_is_zero(self):
        m = self.m
        fwd = m.spot["EUR"] * m.df("EUR", 1.0) / m.df("USD", 1.0)
        self.assertAlmostEqual(FXForward("f", "F", ccy="EUR", notional=1e6, strike=fwd, maturity=1.0).value(m), 0.0, places=6)

    def test_eq_option_put_call_parity_in_market(self):
        m = self.m
        c = EqOption("o", "E", name="EQ_FIN", qty=1.0, strike=95.0, maturity=1.0, call=True).value(m)
        p = EqOption("o", "E", name="EQ_FIN", qty=1.0, strike=95.0, maturity=1.0, call=False).value(m)
        fwd = m.spot["EQ_FIN"] * math.exp(-m.div["EQ_FIN"]) - 95.0 * m.df("USD", 1.0)
        self.assertAlmostEqual(c - p, fwd, places=8)

    def test_exotic_has_no_value_or_exposure(self):
        e = Exotic("x", "E", rrao="exotic", notional=5.0)
        self.assertEqual(e.value(self.m), 0.0)
        self.assertEqual(e.exposures(), [])
        self.assertEqual(e.rrao_notional(self.m), 5.0)

    def test_index_cds_look_through_conserves_notional(self):
        c = CDS("c", "C", name="IG_INDEX", notional=1e6, buyer=True, index=True)
        items = c.drc_items(self.m)
        self.assertEqual(len(items), 25)
        self.assertAlmostEqual(sum(i["notional"] for i in items), -1e6)

    def test_government_bond_has_no_default_risk_item(self):
        self.assertEqual(Bond("b", "R", face=1.0, coupon=0.04, maturity=3).drc_items(self.m), [])


class Sensitivities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = build_market()
        cls.book = build_book()
        cls.sens = sensitivities(cls.book, cls.m)

    def test_book_is_deterministic(self):
        a = [i.iid for i in build_book(7)]
        b = [i.iid for i in build_book(7)]
        self.assertEqual(a, b)
        v1 = sum(i.value(self.m) for i in build_book(7))
        v2 = sum(i.value(self.m) for i in build_book(7))
        self.assertEqual(v1, v2)
        self.assertNotEqual(v1, sum(i.value(self.m) for i in build_book(8)))

    def test_covers_all_four_risk_classes_and_both_measures(self):
        self.assertEqual(set(self.sens[C.RISK_CLASS]), {C.GIRR, C.CSR, C.EQ, C.FX})
        self.assertEqual(set(self.sens[C.MEASURE]), {C.DELTA, C.VEGA})

    def test_equity_option_delta_matches_analytic(self):
        opt = EqOption("o", "E", name="EQ_TECH", qty=1000.0, strike=190, maturity=1.0, call=True)
        s = sensitivities([opt], self.m)
        row = s[(s[C.RISK_CLASS] == C.EQ) & (s[C.MEASURE] == C.DELTA)].iloc[0]
        sp = self.m.spot["EQ_TECH"]
        g = bs_greeks(sp, 190, self.m.zero("USD", 1.0), self.m.div["EQ_TECH"], self.m.vol("vol_eq", "EQ_TECH", 1.0), 1.0)
        analytic = 1000.0 * g["delta"] * sp * 0.01
        gamma_term = 0.5 * 1000.0 * g["gamma"] * (sp * 0.01) ** 2
        self.assertAlmostEqual(row[C.SENSITIVITY] * 0.01, analytic + gamma_term, delta=abs(analytic) * 1e-3)

    def test_vega_sensitivity_equals_vega_times_vol(self):
        opt = EqOption("o", "E", name="EQ_TECH", qty=1000.0, strike=190, maturity=1.0, call=True)
        s = sensitivities([opt], self.m)
        vega_total = s[(s[C.MEASURE] == C.VEGA)][C.SENSITIVITY].sum()
        sig = self.m.vol("vol_eq", "EQ_TECH", 1.0)
        g = bs_greeks(self.m.spot["EQ_TECH"], 190, self.m.zero("USD", 1.0), self.m.div["EQ_TECH"], sig, 1.0)
        self.assertAlmostEqual(vega_total, 1000.0 * g["vega"] * sig, delta=abs(vega_total) * 5e-3)

    def test_irs_tenor_sensitivities_sum_to_parallel_pv01(self):
        irs = IRS("i", "R", notional=1e7, fixed=0.04, maturity=7, payer=True)
        s = sensitivities([irs], self.m)
        total = s[s[C.RISK_CLASS] == C.GIRR][C.SENSITIVITY].sum()
        parallel = (irs.value(self.m.bumped_parallel("girr", "USD", 1e-4)) - irs.value(self.m)) / 1e-4
        self.assertAlmostEqual(total, parallel, delta=abs(parallel) * 2e-3)

    def test_sensitivities_are_additive_across_a_book_split(self):
        half = len(self.book) // 2
        whole = self.sens.groupby([C.RISK_CLASS, C.MEASURE, C.BUCKET, C.RISK_FACTOR, C.CURVE], dropna=False)[C.SENSITIVITY].sum()
        parts = pd.concat([sensitivities(self.book[:half], self.m), sensitivities(self.book[half:], self.m)])
        parts = parts.groupby([C.RISK_CLASS, C.MEASURE, C.BUCKET, C.RISK_FACTOR, C.CURVE], dropna=False)[C.SENSITIVITY].sum()
        diff = (whole - parts.reindex(whole.index).fillna(0)).abs().max()
        self.assertLess(diff, 1e-6)

    def test_stock_is_pure_equity_delta(self):
        s = sensitivities([Stock("s", "E", name="EQ_CONS", shares=1000.0)], self.m)
        self.assertEqual(len(s), 1)
        self.assertAlmostEqual(s[C.SENSITIVITY].iloc[0], 1000.0 * self.m.spot["EQ_CONS"], places=4)  # per unit relative move

    def test_linear_book_has_no_curvature(self):
        linear = [i for i in self.book if not i.is_option]
        self.assertTrue(curvature(linear, self.m).empty)
        res = sa_capital(sensitivities(linear, self.m), curvature(linear, self.m), drc_positions(linear, self.m),
                         rrao_positions(linear, self.m))
        self.assertEqual(res.sbm.by_scenario[res.sbm.by_scenario[C.MEASURE] == C.CURVATURE][list(("low", "medium", "high"))].to_numpy().sum(), 0.0)

    def test_fx_forward_curvature_is_numerically_zero_even_if_forced(self):
        fwd = FXForward("f", "F", ccy="EUR", notional=1e7, strike=1.05, maturity=1.0)
        cv = curvature([fwd], self.m, include_linear=True)
        fx_row = cv[cv[C.RISK_CLASS] == C.FX].iloc[0]
        self.assertAlmostEqual(fx_row[C.CVR_UP], 0.0, places=4)
        self.assertAlmostEqual(fx_row[C.CVR_DOWN], 0.0, places=4)

    def test_long_option_curvature_is_not_a_loss_and_short_option_is(self):
        long_opt = FXOption("o", "F", ccy="EUR", notional=1e7, strike=1.08, maturity=1.0, call=True)
        short_opt = FXOption("o", "F", ccy="EUR", notional=-1e7, strike=1.08, maturity=1.0, call=True)
        cl = curvature([long_opt], self.m)
        cs = curvature([short_opt], self.m)
        fx_l = cl[cl[C.RISK_CLASS] == C.FX].iloc[0]
        fx_s = cs[cs[C.RISK_CLASS] == C.FX].iloc[0]
        self.assertLess(fx_l[C.CVR_UP], 0)
        self.assertGreater(fx_s[C.CVR_UP], 0)
        self.assertAlmostEqual(fx_l[C.CVR_UP], -fx_s[C.CVR_UP])


class StandardisedTotal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        m = build_market()
        book = build_book()
        cls.res = sa_capital(sensitivities(book, m), curvature(book, m), drc_positions(book, m), rrao_positions(book, m))

    def test_total_reconciles_to_components(self):
        self.assertAlmostEqual(self.res.total, self.res.sbm.charge + self.res.drc.total + self.res.rrao)

    def test_decomposition_rows_sum_to_total(self):
        dec = self.res.decomposition
        body = dec[dec[C.RISK_CLASS] != "TOTAL"]
        self.assertAlmostEqual(body["charge"].sum(), self.res.total, places=4)

    def test_sbm_is_max_of_scenario_totals(self):
        self.assertAlmostEqual(self.res.sbm.charge, max(self.res.sbm.totals.values()))

    def test_charges_are_positive(self):
        self.assertGreater(self.res.sbm.charge, 0)
        self.assertGreater(self.res.drc.total, 0)
        self.assertGreater(self.res.rrao, 0)

    def test_rrao_matches_hand_computation(self):
        m = build_market()
        pos_df = rrao_positions(build_book(), m)
        expected = 0.01 * (40e6 + 15e6) + 0.001 * pos_df[pos_df["rrao_type"] == "other"][C.NOTIONAL].sum()
        self.assertAlmostEqual(self.res.rrao, expected, places=4)

    def test_removing_hedges_raises_capital(self):
        m = build_market()
        book = build_book()
        # Dropping every protection-buyer CDS removes the short JTD that the hedge benefit ratio credits.
        no_hedge = [i for i in book if i.iid not in {x.iid for x in book if isinstance(x, CDS) and x.buyer}]
        base = sa_capital(sensitivities(book, m), curvature(book, m), drc_positions(book, m), rrao_positions(book, m))
        alt = sa_capital(sensitivities(no_hedge, m), curvature(no_hedge, m), drc_positions(no_hedge, m),
                         rrao_positions(no_hedge, m))
        self.assertGreater(alt.drc.total, base.drc.total)

    def test_book_value_is_finite(self):
        self.assertTrue(np.isfinite(book_value(build_book(), build_market())))


if __name__ == "__main__":
    unittest.main()
