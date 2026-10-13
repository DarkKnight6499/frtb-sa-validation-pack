"""Synthetic trading book instruments, each revalued from a Market (USD reporting currency)."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .market import Market
from .pricing import bs_price

# Reference data: SBM buckets (MAR21.51, MAR21.72) and credit quality for synthetic names.
CSR_BUCKET = {"SOV_A": 1, "FIN_A": 3, "IND_A": 4, "CNS_A": 5, "TEC_A": 6, "HC_A": 7, "FIN_HY": 11,
              "CNS_HY": 13, "IG_INDEX": 17}
EQ_BUCKET = {"EQ_TECH": 8, "EQ_FIN": 8, "EQ_CONS": 5, "EQ_ENER": 7, "EQ_SMALL": 10, "EQ_INDEX": 12}
NAME_RATING = {"SOV_A": "AA", "FIN_A": "A", "IND_A": "BBB", "CNS_A": "BBB", "TEC_A": "A", "HC_A": "BBB",
               "FIN_HY": "BB", "CNS_HY": "B", "IG_INDEX": "BBB"}
DRC_SECTOR = {"SOV_A": "sovereign"}   # everything else is a corporate
INDEX_CONSTITUENTS = 25               # equal-weight look-through for the synthetic IG index (MAR22.5)


@dataclass
class Instrument:
    iid: str
    desk: str
    rrao: str | None = None
    is_option: bool = False

    def value(self, m: Market) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def exposures(self) -> list[tuple[str, str]]:
        return []

    def rrao_notional(self, m: Market) -> float:
        return 0.0

    def drc_items(self, m: Market) -> list[dict]:
        return []


def _coupon_times(maturity: float) -> list[float]:
    n = int(math.floor(maturity + 1e-9))
    times = [float(i) for i in range(1, n + 1)]
    if maturity - n > 1e-9:
        times.append(maturity)
    return times


@dataclass
class IRS(Instrument):
    ccy: str = "USD"
    notional: float = 0.0       # local currency
    fixed: float = 0.0
    maturity: float = 5.0
    payer: bool = True          # pays fixed

    def value(self, m: Market) -> float:
        times = _coupon_times(self.maturity)
        annuity = sum(m.df(self.ccy, t) for t in times)
        float_leg = 1.0 - m.df(self.ccy, self.maturity)
        pv = self.notional * (float_leg - self.fixed * annuity)
        pv = pv if self.payer else -pv
        return pv * (1.0 if self.ccy == "USD" else m.spot[self.ccy])

    def exposures(self):
        out = [("girr", self.ccy)]
        return out + ([("fx", self.ccy)] if self.ccy != "USD" else [])


@dataclass
class Bond(Instrument):
    ccy: str = "USD"
    face: float = 0.0           # signed: negative is a short position
    coupon: float = 0.0
    maturity: float = 5.0
    issuer: str | None = None   # None for a risk-free (government) bond
    seniority: str = "senior"

    def value(self, m: Market) -> float:
        total = 0.0
        for t in _coupon_times(self.maturity):
            cash = self.coupon + (1.0 if abs(t - self.maturity) < 1e-9 else 0.0)
            spread = m.spread(self.issuer, t) if self.issuer else 0.0
            total += cash * m.df(self.ccy, t) * math.exp(-spread * t)
        return self.face * total

    def exposures(self):
        out = [("girr", self.ccy)]
        return out + ([("csr", self.issuer)] if self.issuer else [])

    def drc_items(self, m: Market) -> list[dict]:
        if self.issuer is None:   # MAR22.7: zero default risk weight on domestic sovereigns, national discretion
            return []
        v = self.value(m)
        return [dict(obligor=self.issuer, seniority=self.seniority, rating=NAME_RATING[self.issuer],
                     drc_bucket=DRC_SECTOR.get(self.issuer, "corporate"), notional=self.face,
                     pnl_to_date=v - self.face, maturity=self.maturity, kind="bond")]


def cds_legs(m: Market, name: str, maturity: float, coupon: float) -> tuple[float, float]:
    """Protection and premium leg PVs per unit notional on a quarterly grid (credit-triangle survival)."""
    lgd = 1.0 - m.recovery
    n = max(int(round(maturity * 4)), 1)
    times = [maturity * (i + 1) / n for i in range(n)]
    prev_t, prev_q = 0.0, 1.0
    prot = prem = 0.0
    for t in times:
        q = math.exp(-m.spread(name, t) * t / lgd)
        d = m.df("USD", t)
        prot += lgd * d * (prev_q - q)
        prem += coupon * (t - prev_t) * d * 0.5 * (prev_q + q)
        prev_t, prev_q = t, q
    return prot, prem


@dataclass
class CDS(Instrument):
    name: str = ""
    notional: float = 0.0
    coupon: float = 0.01
    maturity: float = 5.0
    buyer: bool = True          # protection buyer (short credit)
    index: bool = False

    def value(self, m: Market) -> float:
        prot, prem = cds_legs(m, self.name, self.maturity, self.coupon)
        v = self.notional * (prot - prem)
        return v if self.buyer else -v

    def exposures(self):
        return [("girr", "USD"), ("csr", self.name)]

    def drc_items(self, m: Market) -> list[dict]:
        sign = -1.0 if self.buyer else 1.0
        v = self.value(m)
        if not self.index:
            return [dict(obligor=self.name, seniority="senior", rating=NAME_RATING[self.name],
                         drc_bucket=DRC_SECTOR.get(self.name, "corporate"), notional=sign * self.notional,
                         pnl_to_date=v, maturity=self.maturity, kind="cds")]
        n = INDEX_CONSTITUENTS
        return [dict(obligor=f"{self.name}_{i}", seniority="senior", rating=NAME_RATING[self.name],
                     drc_bucket="corporate", notional=sign * self.notional / n, pnl_to_date=v / n,
                     maturity=self.maturity, kind="index_cds") for i in range(n)]


@dataclass
class FXForward(Instrument):
    ccy: str = "EUR"
    notional: float = 0.0       # signed foreign-currency amount received at maturity
    strike: float = 1.0
    maturity: float = 1.0

    def value(self, m: Market) -> float:
        t = self.maturity
        return self.notional * (m.spot[self.ccy] * m.df(self.ccy, t) - self.strike * m.df("USD", t))

    def exposures(self):
        return [("girr", self.ccy), ("girr", "USD"), ("fx", self.ccy)]


@dataclass
class FXOption(Instrument):
    ccy: str = "EUR"
    notional: float = 0.0       # signed foreign-currency amount
    strike: float = 1.0
    maturity: float = 1.0
    call: bool = True
    is_option: bool = True

    def value(self, m: Market) -> float:
        t = self.maturity
        sigma = m.vol("vol_fx", self.ccy, t)
        return self.notional * bs_price(m.spot[self.ccy], self.strike, m.zero("USD", t), m.zero(self.ccy, t),
                                        sigma, t, self.call)

    def exposures(self):
        return [("girr", self.ccy), ("girr", "USD"), ("fx", self.ccy), ("vol_fx", self.ccy)]


@dataclass
class EqOption(Instrument):
    name: str = ""
    qty: float = 0.0            # signed number of underlying units
    strike: float = 100.0
    maturity: float = 1.0
    call: bool = True
    is_option: bool = True

    def value(self, m: Market) -> float:
        t = self.maturity
        sigma = m.vol("vol_eq", self.name, t)
        return self.qty * bs_price(m.spot[self.name], self.strike, m.zero("USD", t), m.div[self.name],
                                   sigma, t, self.call)

    def exposures(self):
        return [("girr", "USD"), ("eq", self.name), ("vol_eq", self.name)]

    def rrao_notional(self, m: Market) -> float:
        return abs(self.qty) * m.spot[self.name] if self.rrao else 0.0


@dataclass
class Stock(Instrument):
    name: str = ""
    shares: float = 0.0

    def value(self, m: Market) -> float:
        return self.shares * m.spot[self.name]

    def exposures(self):
        return [("eq", self.name)]

    def drc_items(self, m: Market) -> list[dict]:
        v = self.value(m)
        return [dict(obligor=self.name, seniority="equity", rating="UNRATED", drc_bucket="corporate",
                     notional=v, pnl_to_date=0.0, maturity=1.0, kind="equity")]


@dataclass
class Exotic(Instrument):
    """Instrument with an exotic underlying: no SBM or DRC risk factors, only the RRAO (MAR23.3)."""
    notional: float = 0.0

    def value(self, m: Market) -> float:
        return 0.0

    def rrao_notional(self, m: Market) -> float:
        return abs(self.notional)
