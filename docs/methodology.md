# Methodology

Source for rules: Basel Framework market risk chapters MAR20 to MAR33 as published in BCBS d457 (January 2019). Paragraph references sit beside each constant in the code. Anything marked stylized is an implementation choice, not a rule.

## 1. Synthetic book

`portfolio.build_book` creates 46 instruments from a seeded generator (`numpy.random.default_rng`): swaps, bonds, CDS, stocks, equity options, FX forwards and options and exotics. `sensitivities` revalues each instrument under bumps: GIRR 1 bp per tenor, equity and FX 1% relative, CSR 1 bp, vol 1% relative (so the vega sensitivity is vega times sigma). Gamma is a diagonal second difference used for IMA delta-gamma P&L. Curvature positions come from the MAR21.5 up and down shocks with the risk weight as the shock size, for option instruments only unless `include_linear=True`.

## 2. Standardised approach

- Delta and vega: weighted sensitivities, intra-bucket correlation, bucket capital `K_b = sqrt(max(0, sum WS^2 + sum rho WS_k WS_l))`, with the alternative specification of MAR21.6 when the quantity under the root is negative, then inter-bucket aggregation with gamma.
- Curvature: per risk factor `CVR = -min(V(up) - V - RW*delta, V(down) - V + RW*delta)` style positions (see `curvature_bucket`), bucket aggregation with squared correlations, floor at zero per MAR21.5.
- Three scenarios: correlations multiplied by 1.25 (capped at 1), 1.0 and 0.75 (the low scenario is `max(2*rho - 1, 0.75*rho)` for delta and vega, per MAR21.6). The reported capital for delta, vega and curvature is the maximum across scenarios for the sum of the three, as MAR21.4 states, and the binding scenario is shown.
- DRC: jump-to-default by position, risk weights by credit quality, LGD by seniority, maturity weighting with a three month floor, netting within obligor, hedge benefit ratio across buckets. RRAO: 1% of notional for exotic underlyings, 0.1% for other residual risks.
- Total SA is SBM plus DRC plus RRAO. `capital_decomposition` presents the components by risk class and measure under each scenario.

## 3. Internal models approach

- P&L: delta-gamma over a synthetic correlated Student-t panel of risk factor changes (2,500 days) with a planted high-volatility window (stylized).
- Liquidity horizons: factors are assigned to the MAR33.12 categories (10, 20, 40, 60, 120 days), with the assignment of synthetic names being stylized. ES is computed by the cascade `ES = sqrt(ES_T(P)^2 + sum_j (ES_T(P,j) sqrt((LH_j - LH_{j-1})/T))^2)` with T = 10.
- Reduced set: a rule that picks factors until a share of the full ES is reached, per class, then tops up (stylized). The stress window is the 250 day window maximising ES_R,S; ES = ES_R,S * max(ES_F,C / ES_R,C, 1). IMCC = rho * ES_unconstrained + (1 - rho) * sum of class ES with rho = 0.5.
- RFET: 24 observations in the year with no 90 day gap containing fewer than 4, or 100 observations. Failing factors go to NMRF. The stress scenario charge is a simplified calibration with horizon at least 20 days and SES aggregation with rho = 0.6 across non-idiosyncratic factors.
- Backtesting: exceptions against one day VaR at 99% and 97.5%, green, amber and red zones by cumulative binomial probability (amber from 95% cumulative probability, red from 99.99%), and a Kupiec test.

## 4. P&L attribution

Hypothetical and risk-theoretical P&L per desk are generated from the synthetic panel with desk-specific model error. Spearman rank correlation and the Kolmogorov-Smirnov statistic (p-value by the asymptotic series with lambda = sqrt(n/2) * D) give zones: green, amber, red, with the thresholds of MAR32. Desk outcome: green IMA eligible, amber IMA with surcharge, red standardised.

## 5. Validation of the pricer

The model under test is a flat-volatility Black-Scholes-Merton closed form for European options and a 500 step CRR tree for American puts.

- Challengers: Monte Carlo with antithetic variates and a regression control variate on the discounted terminal price; Crank-Nicolson finite differences in log price with Rannacher start-up (two implicit half steps first) and Brennan-Schwartz projection for early exercise on puts.
- Conceptual checks: parity, bounds, monotonicity, convexity in strike, Greek bounds, delta against finite differences, implied-vol round trip, expiry limit, early exercise ordering, tree probability range.
- Benchmark grid: strikes 80 to 120, maturities 0.25 to 2 years, calls and puts, against both challengers; American puts against the PDE and a 5,000 step tree.
- Convergence: fitted log-log slopes for tree error (about -1), Monte Carlo error (about -0.5) and the FD grid (about -2).
- Outcomes: implied-vol error against a stated synthetic smile with a limit in a moneyness band, plus a delta-hedging backtest at model volatility, weekly rebalancing and a 30% realised volatility shock.
- Findings carry severities (High, Medium, Low, Info) tied to rules in `validation/run.py`, and the conclusion follows the worst open severity. Monitoring metrics have Green, Amber and Red thresholds, which are policy choices for this exercise.

## 6. Out of scope and future work

A regulatory assistant that answers questions over the Basel text and the SR 26-2 guidance using retrieval and a language model is a possible extension and is not implemented. Securitisation and CTP treatments, commodities and jurisdictional add-ons are also not implemented.
