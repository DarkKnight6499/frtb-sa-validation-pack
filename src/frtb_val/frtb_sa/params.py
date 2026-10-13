"""Basel FRTB standardised approach parameters.

Source: BCBS d457 (January 2019) "Minimum capital requirements for market risk", whose chapters
are MAR20 to MAR23 of the Basel Framework, read from the bis.org PDF. Each constant cites its
paragraph. Items marked STYLIZED are simplifications that are not taken from the text.
"""
from __future__ import annotations

import math

import numpy as np

# MAR21.8 / MAR21.42 Table 1: GIRR delta risk weights (decimal), by tenor in years.
GIRR_TENORS = (0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30)
GIRR_RISK_WEIGHTS = (0.017, 0.017, 0.016, 0.013, 0.012, 0.011, 0.011, 0.011, 0.011, 0.011)
GIRR_INFLATION_XCCY_RW = 0.016          # MAR21.43
GIRR_THETA = 0.03                       # MAR21.46 footnote 13: rho = max(exp(-theta |Tk-Tl| / min), 40%)
GIRR_RHO_FLOOR = 0.40
GIRR_RHO_SAME_TENOR_DIFF_CURVE = 0.999  # MAR21.45
GIRR_RHO_INFLATION = 0.40               # MAR21.48
GIRR_RHO_XCCY = 0.0                     # MAR21.49
GIRR_GAMMA = 0.50                       # MAR21.50
GIRR_CURVATURE_RW = 0.017               # MAR21.99: highest GIRR delta weight shifts every tenor in parallel

# MAR21.87 / MAR21.89: FX
FX_RW = 0.15
FX_GAMMA = 0.60

# MAR21.53 Table 4: CSR non-securitisation delta risk weights, buckets 1..18.
CSR_RISK_WEIGHTS = {1: 0.005, 2: 0.010, 3: 0.050, 4: 0.030, 5: 0.030, 6: 0.020, 7: 0.015, 8: 0.025,
                    9: 0.020, 10: 0.040, 11: 0.120, 12: 0.070, 13: 0.085, 14: 0.055, 15: 0.050,
                    16: 0.120, 17: 0.015, 18: 0.050}
CSR_RHO_NAME = 0.35          # MAR21.54(1), buckets 1-15
CSR_RHO_NAME_INDEX = 0.80    # MAR21.55(1), buckets 17-18
CSR_RHO_TENOR = 0.65         # MAR21.54(2)
CSR_RHO_BASIS = 0.999        # MAR21.54(3)
CSR_RATING_GAMMA = 0.50      # MAR21.57(1): IG versus HY/NR across buckets 1-15
# MAR21.57(2) Table 5: sector gamma between distinct sectors. Keys are sector ids; buckets 1-8 are IG
# sectors and bucket b+8 (9-15) is the HY/NR twin of sector b. Sector 8 is covered bonds. 16 = other.
_T5 = {
    (1, 2): .75, (1, 3): .10, (1, 4): .20, (1, 5): .25, (1, 6): .20, (1, 7): .15, (1, 8): .10,
    (2, 3): .05, (2, 4): .15, (2, 5): .20, (2, 6): .15, (2, 7): .10, (2, 8): .10,
    (3, 4): .05, (3, 5): .15, (3, 6): .20, (3, 7): .05, (3, 8): .20,
    (4, 5): .20, (4, 6): .25, (4, 7): .05, (4, 8): .05,
    (5, 6): .25, (5, 7): .05, (5, 8): .15,
    (6, 7): .05, (6, 8): .20,
    (7, 8): .05,
}
CSR_SECTOR_GAMMA_TABLE = {**_T5, **{(b, a): v for (a, b), v in _T5.items()}}
CSR_INDEX_GAMMA = 0.45        # Table 5 columns 17 and 18 versus buckets 1-15 and 8
CSR_INDEX_INDEX_GAMMA = 0.75  # Table 5, bucket 17 versus 18
CSR_OTHER_BUCKET = 16

# MAR21.77 Table 10: equity spot risk weights, buckets 1..13.
EQ_RISK_WEIGHTS = {1: 0.55, 2: 0.60, 3: 0.45, 4: 0.55, 5: 0.30, 6: 0.35, 7: 0.40, 8: 0.50, 9: 0.70,
                   10: 0.50, 11: 0.70, 12: 0.15, 13: 0.25}
# MAR21.78(2): equity intra-bucket correlation between different issuers, spot to spot.
EQ_RHO = {**{b: 0.15 for b in (1, 2, 3, 4)}, **{b: 0.25 for b in (5, 6, 7, 8)}, 9: 0.075, 10: 0.125,
          12: 0.80, 13: 0.80}
EQ_RHO_SPOT_REPO_SAME_NAME = 0.999   # MAR21.78(1)
EQ_OTHER_BUCKET = 11
EQ_LARGE_CAP_VEGA_BUCKETS = (1, 2, 3, 4, 5, 6, 7, 8, 12, 13)  # MAR21.92 Table 13: large cap and indices

# MAR21.92 Table 13 and footnote 24: vega risk weight = min(RW_sigma * sqrt(LH / 10), 100%), RW_sigma = 55%.
VEGA_RW_SIGMA = 0.55
VEGA_LIQUIDITY_HORIZON = {"GIRR": 60, "CSR": 120, "EQ_LARGE": 20, "EQ_SMALL": 60, "FX": 40}
VEGA_ALPHA = 0.01        # MAR21.93 / MAR21.94: option-maturity correlation exp(-alpha |Tk-Tl| / min(Tk,Tl))

# MAR21.6 correlation scenarios.
HIGH_MULT = 1.25
LOW_FLOOR_MULT = 0.75


def vega_risk_weight(liquidity_horizon: float) -> float:
    return min(VEGA_RW_SIGMA * math.sqrt(liquidity_horizon / 10.0), 1.0)


def girr_tenor_rho(tk: float, tl: float) -> float:
    """MAR21.46 footnote 13: max(exp(-theta |Tk-Tl| / min(Tk,Tl)), 40%)."""
    return max(math.exp(-GIRR_THETA * abs(tk - tl) / min(tk, tl)), GIRR_RHO_FLOOR)


def vega_maturity_rho(tk: float, tl: float) -> float:
    """MAR21.93: min(exp(-alpha |Tk-Tl| / min(Tk,Tl)), 1)."""
    return min(math.exp(-VEGA_ALPHA * abs(tk - tl) / min(tk, tl)), 1.0)


def csr_sector(bucket: int) -> int | None:
    """Sector id 1..8 for buckets 1-15 (HY twin maps to the same sector), None for 16-18."""
    if 1 <= bucket <= 8:
        return bucket
    if 9 <= bucket <= 15:
        return bucket - 8
    return None


def csr_gamma(b: int, c: int) -> float:
    """MAR21.57 inter-bucket CSR non-securitisation correlation."""
    if b == c:
        return 1.0
    if CSR_OTHER_BUCKET in (b, c):
        return 0.0
    if b in (17, 18) and c in (17, 18):
        return CSR_INDEX_INDEX_GAMMA
    if b in (17, 18) or c in (17, 18):
        return CSR_INDEX_GAMMA
    sb, sc = csr_sector(b), csr_sector(c)
    rating = CSR_RATING_GAMMA if ((b <= 8) != (c <= 8)) and b <= 15 and c <= 15 else 1.0
    sector = 1.0 if sb == sc else CSR_SECTOR_GAMMA_TABLE[(sb, sc)]
    return rating * sector


def eq_gamma(b: int, c: int) -> float:
    """MAR21.80 inter-bucket equity correlation."""
    if b == c:
        return 1.0
    if EQ_OTHER_BUCKET in (b, c):
        return 0.0
    if {b, c} == {12, 13}:
        return 0.75
    if b <= 10 and c <= 10:
        return 0.15
    return 0.45


# MAR22.12 LGD, MAR22.24 Table 2 default risk weights, MAR22.15/22.18 maturity weighting.
DRC_LGD = {"senior": 0.75, "non_senior": 1.0, "equity": 1.0, "covered": 0.25}
DRC_RISK_WEIGHTS = {"AAA": 0.005, "AA": 0.02, "A": 0.03, "BBB": 0.06, "BB": 0.15, "B": 0.30,
                    "CCC": 0.50, "UNRATED": 0.15, "DEFAULTED": 1.0}
DRC_MIN_MATURITY_WEIGHT = 0.25
DRC_SENIORITY_RANK = {"senior": 1, "non_senior": 2, "equity": 3}  # MAR22.19(1): shorts offset equal/lower seniority

# MAR23.8: residual risk add-on risk weights on gross notional.
RRAO_RW = {"exotic": 0.01, "other": 0.001}

# Vega/delta correlation scenario helpers are in sbm.py.
