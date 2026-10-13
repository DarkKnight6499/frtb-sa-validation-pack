# Independent Model Validation Report: Vanilla Option Pricer

Structured after the sections of the Supervisory Guidance on Model Risk Management issued April 17, 2026 by the Federal Reserve (SR 26-2), the OCC and the FDIC, which superseded SR 11-7: model risk and materiality (section III), development and use (IV), validation through conceptual soundness, outcomes analysis and ongoing monitoring (V), and governance and controls (VI). The guidance is non-prescriptive, so this report is a paraphrase of structure and not a compliance artefact. All figures below were produced by `examples/run_validation.py` on synthetic inputs with fixed seeds.

## 1. Executive Summary

**Conclusion: Not approved: at least one High finding is open.**

Findings by severity: High 1, Medium 1, Low 2, Info 5.

The closed-form European prices agree with the Monte Carlo challenger (max |z| 2.00) and the finite-difference challenger (max difference 0.0004). The American put tree agrees with the PDE challenger to 0.0025. The model's main weakness is the flat volatility assumption: against a synthetic skewed market the implied-vol error reaches 7.53 vol points and 4 of 8 in-band quotes breach the 1.0 point limit.

## 2. Model Description

Model under validation: a front-office pricer with one flat volatility (20%), a constant risk-free rate (5.0%) and a continuous dividend yield (2.0%).

- European calls and puts: Black-Scholes-Merton closed form.
- American puts: Cox-Ross-Rubinstein binomial tree with 500 steps and early exercise at every node.

Challengers built independently for this validation:

- Monte Carlo with antithetic draws and a regression control variate (the discounted terminal price), 200,000 paths.
- Crank-Nicolson finite differences in log price with Rannacher start-up and Brennan-Schwartz projection for early exercise, 800 space and 800 time steps.

Statement of purpose (section IV expects development to begin with one): intended use is not documented for this synthetic model, so every strike and maturity on the test grid is treated as in scope, and that gap is itself recorded as a finding.

Materiality and proportionate scope (section III). The ratings below are assumptions of this exercise, not observations of a bank.

| Factor | Assessment | Basis |
|---|---|---|
| Inherent risk | Medium | Few assumptions, but one (flat volatility) drives the result; constant rates and yield |
| Model exposure | Assumed large | Stand-in for an options book priced and hedged off this model; no real portfolio |
| Model purpose | Higher | Pricing and hedging of financial risk exposures, which the guidance treats as higher risk |
| Materiality and validation depth | High: full scope | Conceptual soundness, challenger models, outcomes analysis and ongoing monitoring all performed |

## 3. Conceptual Soundness

SR 26-2 section V asks for an assessment and documentation of model design, assumptions and data selection, and allows benchmarking where a theory review is impractical. Property checks:

| Check | Passed | Detail |
|---|---|---|
| European put-call parity | yes | max residual 1.78e-14 |
| No-arbitrage price bounds | yes | 0 violations on 40 prices |
| Monotonicity in spot (call up, put down) | yes | 41 spot points |
| Monotonicity in volatility | yes | 30 vol points |
| Convexity in strike (butterfly >= 0) | yes | 41 strike points |
| Greek bounds (delta range, gamma and vega non-negative, call/put equalities) | yes | 20 grid points |
| Analytic delta equals finite difference of price | yes | max error 4.22e-08 |
| Implied volatility round trip | yes | max error 1.31e-09 |
| Expiry limit: price tends to intrinsic as t tends to 0 | yes | gap at t=1e-6 is 7.98e-03 |
| American put >= European put (same tree) and >= intrinsic | yes | early-exercise premium from 0.0002 to 2.7075 |
| American call equals European call when q = 0 | yes | max difference 0.00e+00 |
| Tree risk-neutral probability inside (0, 1) for tested steps and maturities | yes | 0 violations |

Assumption register:

| Assumption | Assessment |
|---|---|
| Lognormal diffusion, constant volatility across strike and maturity | Limitation: contradicted by any skew or term structure; quantified in outcomes analysis |
| Constant risk-free rate and continuous dividend yield | Acceptable for short-dated vanilla options; discrete dividends not supported |
| European exercise by closed form; American put by CRR tree with early exercise on nodes | Reasonable; tree discretisation error measured against an independent PDE solution |
| Frictionless continuous hedging | Limitation: daily discrete hedging error measured in outcomes analysis |
| American calls are priced by the same tree | Not benchmarked against a challenger: the PDE challenger supports American puts only |

## 4. Developmental Evidence

### 4.1 Benchmark grid (strikes 80 to 120, maturities 0.25 to 2 years, spot 100)

European prices: 40 price comparisons. Monte Carlo z-score is (front office minus challenger) over the challenger standard error; the variance reduction from the control variate averages 20.4x.

| type | strike | maturity | front_office | mc_cv | mc_stderr | z_score | fd | fo_minus_fd |
|---|---|---|---|---|---|---|---|---|
| call | 80 | 0.25 | 20.5268 | 20.5279 | 0.0007 | -1.5916 | 20.5269 | -0.0000 |
| put | 80 | 0.25 | 0.0318 | 0.0329 | 0.0007 | -1.5916 | 0.0319 | -0.0000 |
| call | 80 | 0.50 | 21.2161 | 21.2158 | 0.0015 | 0.2183 | 21.2161 | -0.0000 |
| put | 80 | 0.50 | 0.2359 | 0.2356 | 0.0015 | 0.2183 | 0.2359 | -0.0000 |
| call | 80 | 1.00 | 22.7641 | 22.7644 | 0.0018 | -0.1286 | 22.7642 | -0.0000 |
| put | 80 | 1.00 | 0.8426 | 0.8428 | 0.0018 | -0.1286 | 0.8427 | -0.0000 |
| call | 80 | 2.00 | 25.6408 | 25.6401 | 0.0027 | 0.2581 | 25.6408 | -0.0000 |
| put | 80 | 2.00 | 1.9489 | 1.9481 | 0.0027 | 0.2581 | 1.9489 | -0.0000 |
| call | 90 | 0.25 | 11.2284 | 11.2279 | 0.0011 | 0.4530 | 11.2284 | -0.0000 |
| put | 90 | 0.25 | 0.6091 | 0.6087 | 0.0011 | 0.4530 | 0.6091 | -0.0000 |
| call | 90 | 0.50 | 12.6719 | 12.6723 | 0.0017 | -0.2002 | 12.6718 | 0.0001 |
| put | 90 | 0.50 | 1.4448 | 1.4452 | 0.0017 | -0.2002 | 1.4447 | 0.0001 |
| call | 90 | 1.00 | 15.1237 | 15.1227 | 0.0033 | 0.2989 | 15.1236 | 0.0001 |
| put | 90 | 1.00 | 2.7145 | 2.7135 | 0.0033 | 0.2989 | 2.7144 | 0.0001 |
| call | 90 | 2.00 | 18.9469 | 18.9529 | 0.0056 | -1.0630 | 18.9469 | -0.0000 |
| put | 90 | 2.00 | 4.3033 | 4.3093 | 0.0056 | -1.0630 | 4.3033 | -0.0000 |
| call | 100 | 0.25 | 4.3359 | 4.3335 | 0.0033 | 0.6989 | 4.3356 | 0.0002 |
| put | 100 | 0.25 | 3.5924 | 3.5901 | 0.0033 | 0.6989 | 3.5922 | 0.0002 |
| call | 100 | 0.50 | 6.3076 | 6.3148 | 0.0046 | -1.5725 | 6.3074 | 0.0003 |
| put | 100 | 0.50 | 4.8336 | 4.8408 | 0.0046 | -1.5725 | 4.8334 | 0.0003 |
| call | 100 | 1.00 | 9.2270 | 9.2291 | 0.0064 | -0.3242 | 9.2267 | 0.0003 |
| put | 100 | 1.00 | 6.3301 | 6.3321 | 0.0064 | -0.3242 | 6.3298 | 0.0003 |
| call | 100 | 2.00 | 13.5218 | 13.5137 | 0.0086 | 0.9381 | 13.5214 | 0.0004 |
| put | 100 | 2.00 | 7.9266 | 7.9185 | 0.0086 | 0.9381 | 7.9262 | 0.0004 |
| call | 110 | 0.25 | 1.0859 | 1.0861 | 0.0013 | -0.1549 | 1.0859 | -0.0000 |
| put | 110 | 0.25 | 10.2182 | 10.2184 | 0.0013 | -0.1549 | 10.2182 | -0.0000 |
| call | 110 | 0.50 | 2.5859 | 2.5818 | 0.0020 | 1.9978 | 2.5858 | 0.0001 |
| put | 110 | 0.50 | 10.8650 | 10.8610 | 0.0020 | 1.9978 | 10.8649 | 0.0001 |
| call | 110 | 1.00 | 5.1886 | 5.1826 | 0.0041 | 1.4387 | 5.1886 | -0.0000 |
| put | 110 | 1.00 | 11.8040 | 11.7980 | 0.0041 | 1.4387 | 11.8040 | -0.0000 |
| call | 110 | 2.00 | 9.3579 | 9.3460 | 0.0073 | 1.6482 | 9.3580 | -0.0000 |
| put | 110 | 2.00 | 12.8111 | 12.7991 | 0.0073 | 1.6482 | 12.8111 | -0.0000 |
| call | 120 | 0.25 | 0.1762 | 0.1756 | 0.0017 | 0.3630 | 0.1763 | -0.0001 |
| put | 120 | 0.25 | 19.1843 | 19.1837 | 0.0017 | 0.3630 | 19.1844 | -0.0001 |
| call | 120 | 0.50 | 0.8825 | 0.8824 | 0.0026 | 0.0620 | 0.8826 | -0.0000 |
| put | 120 | 0.50 | 18.9147 | 18.9146 | 0.0026 | 0.0620 | 18.9148 | -0.0000 |
| call | 120 | 1.00 | 2.7118 | 2.7111 | 0.0027 | 0.2584 | 2.7118 | 0.0000 |
| put | 120 | 1.00 | 18.8394 | 18.8387 | 0.0027 | 0.2584 | 18.8394 | 0.0000 |
| call | 120 | 2.00 | 6.3086 | 6.3119 | 0.0042 | -0.7959 | 6.3084 | 0.0002 |
| put | 120 | 2.00 | 18.8101 | 18.8134 | 0.0042 | -0.7959 | 18.8099 | 0.0002 |

American put, tree against PDE:

| strike | maturity | front_office_tree | fd_challenger | tree_5000 | tree_minus_fd | tree_minus_ref | early_exercise_premium |
|---|---|---|---|---|---|---|---|
| 80 | 0.25 | 0.0318 | 0.0321 | 0.0320 | -0.0003 | -0.0002 | -0.0000 |
| 80 | 0.50 | 0.2394 | 0.2393 | 0.2393 | 0.0001 | 0.0001 | 0.0035 |
| 80 | 1.00 | 0.8686 | 0.8676 | 0.8677 | 0.0010 | 0.0009 | 0.0260 |
| 80 | 2.00 | 2.0699 | 2.0692 | 2.0692 | 0.0007 | 0.0007 | 0.1210 |
| 90 | 0.25 | 0.6159 | 0.6157 | 0.6157 | 0.0002 | 0.0002 | 0.0068 |
| 90 | 0.50 | 1.4757 | 1.4744 | 1.4747 | 0.0013 | 0.0010 | 0.0309 |
| 90 | 1.00 | 2.8216 | 2.8214 | 2.8219 | 0.0002 | -0.0003 | 0.1071 |
| 90 | 2.00 | 4.6374 | 4.6368 | 4.6370 | 0.0006 | 0.0003 | 0.3340 |
| 100 | 0.25 | 3.6539 | 3.6548 | 3.6550 | -0.0009 | -0.0011 | 0.0614 |
| 100 | 0.50 | 4.9754 | 4.9766 | 4.9768 | -0.0012 | -0.0014 | 0.1418 |
| 100 | 1.00 | 6.6588 | 6.6603 | 6.6605 | -0.0014 | -0.0017 | 0.3287 |
| 100 | 2.00 | 8.6875 | 8.6893 | 8.6896 | -0.0018 | -0.0021 | 0.7609 |
| 110 | 0.25 | 10.5084 | 10.5075 | 10.5077 | 0.0008 | 0.0007 | 0.2901 |
| 110 | 0.50 | 11.3287 | 11.3267 | 11.3269 | 0.0020 | 0.0017 | 0.4637 |
| 110 | 1.00 | 12.6143 | 12.6118 | 12.6121 | 0.0025 | 0.0023 | 0.8104 |
| 110 | 2.00 | 14.3283 | 14.3268 | 14.3271 | 0.0015 | 0.0013 | 1.5172 |
| 120 | 0.25 | 20.0000 | 20.0000 | 20.0000 | 0.0000 | 0.0000 | 0.8157 |
| 120 | 0.50 | 20.0472 | 20.0470 | 20.0471 | 0.0002 | 0.0001 | 1.1324 |
| 120 | 1.00 | 20.5085 | 20.5082 | 20.5085 | 0.0003 | 0.0001 | 1.6691 |
| 120 | 2.00 | 21.5178 | 21.5169 | 21.5173 | 0.0009 | 0.0005 | 2.7077 |

### 4.2 Convergence

Binomial tree, ATM one-year European put against the closed form (fitted log-log slope -1.00; step-parity oscillation 0.00061):

| steps | price | abs_error |
|---|---|---|
| 10 | 6.138401 | 0.191680 |
| 20 | 6.233525 | 0.096556 |
| 40 | 6.281637 | 0.048444 |
| 80 | 6.305819 | 0.024262 |
| 160 | 6.317940 | 0.012141 |
| 320 | 6.324008 | 0.006073 |
| 640 | 6.327044 | 0.003037 |
| 1280 | 6.328562 | 0.001519 |
| 2560 | 6.329321 | 0.000759 |

Monte Carlo, standard error against paths (fitted slope -0.51, about -0.5 expected):

| paths | price | abs_error | stderr_cv | stderr_plain | variance_reduction |
|---|---|---|---|---|---|
| 1024 | 6.217122 | 0.112959 | 0.095604 | 0.216455 | 5.126096 |
| 4096 | 6.369606 | 0.039525 | 0.044397 | 0.102366 | 5.316179 |
| 16384 | 6.286463 | 0.043617 | 0.023078 | 0.052405 | 5.156524 |
| 65536 | 6.316336 | 0.013745 | 0.011262 | 0.025872 | 5.277435 |
| 262144 | 6.326964 | 0.003117 | 0.005571 | 0.012940 | 5.395239 |

Finite-difference grid, American put against an 8000-step tree (6.660572); fitted slope over the first five grids -2.05:

| grid | price | abs_error |
|---|---|---|
| 50 | 6.564307 | 0.096265 |
| 100 | 6.636380 | 0.024192 |
| 200 | 6.654440 | 0.006132 |
| 400 | 6.659058 | 0.001514 |
| 800 | 6.660257 | 0.000315 |
| 1600 | 6.660568 | 0.000004 |

### 4.3 Greeks comparison

| model | greek | strike | maturity | front_office | challenger | abs_diff | rel_diff |
|---|---|---|---|---|---|---|---|
| European call | delta | 90 | 0.50 | 0.81350 | 0.81384 | 0.00034 | 0.00041 |
| European call | gamma | 90 | 0.50 | 0.01826 | 0.01832 | 0.00006 | 0.00333 |
| European call | vega | 90 | 0.50 | 18.26180 | 18.21197 | 0.04983 | 0.00273 |
| American put | delta | 90 | 0.50 | -0.18118 | -0.18115 | 0.00003 | 0.00016 |
| American put | gamma | 90 | 0.50 | 0.01894 | 0.01893 | 0.00001 | 0.00065 |
| European call | delta | 90 | 1.00 | 0.76589 | 0.76591 | 0.00001 | 0.00002 |
| European call | gamma | 90 | 1.00 | 0.01446 | 0.01450 | 0.00004 | 0.00272 |
| European call | vega | 90 | 1.00 | 28.91963 | 28.89937 | 0.02026 | 0.00070 |
| American put | delta | 90 | 1.00 | -0.22550 | -0.22550 | 0.00000 | 0.00000 |
| American put | gamma | 90 | 1.00 | 0.01560 | 0.01558 | 0.00001 | 0.00095 |
| European call | delta | 100 | 0.50 | 0.56448 | 0.56423 | 0.00025 | 0.00045 |
| European call | gamma | 100 | 0.50 | 0.02750 | 0.02723 | 0.00026 | 0.00953 |
| European call | vega | 100 | 0.50 | 27.49579 | 27.48611 | 0.00968 | 0.00035 |
| American put | delta | 100 | 0.50 | -0.44350 | -0.44342 | 0.00008 | 0.00019 |
| American put | gamma | 100 | 0.50 | 0.02956 | 0.02952 | 0.00004 | 0.00131 |
| European call | delta | 100 | 1.00 | 0.58685 | 0.58655 | 0.00030 | 0.00052 |
| European call | gamma | 100 | 1.00 | 0.01895 | 0.01881 | 0.00015 | 0.00767 |
| European call | vega | 100 | 1.00 | 37.90116 | 37.88844 | 0.01271 | 0.00034 |
| American put | delta | 100 | 1.00 | -0.42311 | -0.42302 | 0.00010 | 0.00023 |
| American put | gamma | 100 | 1.00 | 0.02151 | 0.02148 | 0.00003 | 0.00126 |
| European call | delta | 110 | 0.50 | 0.30646 | 0.30617 | 0.00028 | 0.00092 |
| European call | gamma | 110 | 0.50 | 0.02468 | 0.02501 | 0.00033 | 0.01343 |
| European call | vega | 110 | 0.50 | 24.68190 | 24.66351 | 0.01839 | 0.00074 |
| American put | delta | 110 | 0.50 | -0.73058 | -0.73059 | 0.00001 | 0.00001 |
| American put | gamma | 110 | 0.50 | 0.02895 | 0.02894 | 0.00001 | 0.00022 |
| European call | delta | 110 | 1.00 | 0.40226 | 0.40201 | 0.00025 | 0.00062 |
| European call | gamma | 110 | 1.00 | 0.01906 | 0.01860 | 0.00045 | 0.02386 |
| European call | vega | 110 | 1.00 | 38.11352 | 38.09963 | 0.01389 | 0.00036 |
| American put | delta | 110 | 1.00 | -0.64104 | -0.64103 | 0.00002 | 0.00003 |
| American put | gamma | 110 | 1.00 | 0.02370 | 0.02370 | 0.00001 | 0.00032 |

## 5. Outcomes Analysis

### 5.1 Model prices against a synthetic smile market

The market is generated from a stated smile, `sigma(K, T) = 0.20 + 0.02 exp(-T) - 0.15 m + 0.40 m^2` with `m = ln(K/F)`, plus 0.2 vol point of quote noise. Out-of-the-money options are quoted. The model reprices each quote at its flat 20% volatility and the table gives the implied-vol error.

| type | strike | maturity | market_vol | model_vol | vol_error_pts | market_price | model_price | in_band | breach |
|---|---|---|---|---|---|---|---|---|---|
| put | 80 | 0.25 | 0.2707 | 0.2000 | -7.0661 | 0.2168 | 0.0318 | no | no |
| put | 80 | 0.50 | 0.2711 | 0.2000 | -7.1065 | 0.8630 | 0.2359 | no | no |
| put | 80 | 1.00 | 0.2722 | 0.2000 | -7.2176 | 2.2207 | 0.8426 | no | no |
| put | 80 | 2.00 | 0.2753 | 0.2000 | -7.5302 | 4.3811 | 1.9489 | no | no |
| put | 90 | 0.25 | 0.2391 | 0.2000 | -3.9135 | 1.0401 | 0.6091 | no | no |
| put | 90 | 0.50 | 0.2365 | 0.2000 | -3.6489 | 2.1495 | 1.4448 | no | no |
| put | 90 | 1.00 | 0.2366 | 0.2000 | -3.6557 | 3.8089 | 2.7145 | no | no |
| put | 90 | 2.00 | 0.2390 | 0.2000 | -3.8993 | 5.9699 | 4.3033 | no | no |
| put | 100 | 0.25 | 0.2190 | 0.2000 | -1.9048 | 3.9676 | 3.5924 | yes | yes |
| put | 100 | 0.50 | 0.2126 | 0.2000 | -1.2595 | 5.1800 | 4.8336 | yes | yes |
| put | 100 | 1.00 | 0.2158 | 0.2000 | -1.5770 | 6.9280 | 6.3301 | yes | yes |
| put | 100 | 2.00 | 0.2156 | 0.2000 | -1.5551 | 8.7192 | 7.9266 | yes | yes |
| call | 110 | 0.25 | 0.2043 | 0.2000 | -0.4289 | 1.1468 | 1.0859 | yes | no |
| call | 110 | 0.50 | 0.2040 | 0.2000 | -0.3984 | 2.6846 | 2.5859 | yes | no |
| call | 110 | 1.00 | 0.2002 | 0.2000 | -0.0157 | 5.1946 | 5.1886 | yes | no |
| call | 110 | 2.00 | 0.1944 | 0.2000 | 0.5583 | 9.0553 | 9.3579 | yes | no |
| call | 120 | 0.25 | 0.2028 | 0.2000 | -0.2768 | 0.1895 | 0.1762 | no | no |
| call | 120 | 0.50 | 0.1971 | 0.2000 | 0.2940 | 0.8388 | 0.8825 | no | no |
| call | 120 | 1.00 | 0.1933 | 0.2000 | 0.6710 | 2.5030 | 2.7118 | no | no |
| call | 120 | 2.00 | 0.1891 | 0.2000 | 1.0861 | 5.7470 | 6.3086 | no | no |

Limit test: inside the band |ln(K/F)| <= 0.10 the absolute vol error must stay within 1.0 point for at least 95% of quotes. Result: 4 breaches in 8 in-band quotes (50%), RMSE 1.16 points in band and 3.73 over all quotes. **Limit failed.**

### 5.2 Delta-hedging backtest

One-year at-the-money call sold at the model price and delta hedged at the model volatility, 250 steps, 2,000 simulated paths. Error is terminal hedge value minus payoff.

| scenario | mean_error | std_error | std_over_premium | p01_error | p99_error | limit_passed |
|---|---|---|---|---|---|---|
| realised vol = model vol, daily hedge | 0.0123 | 0.4412 | 0.0478 | -1.1068 | 1.2658 | yes |
| realised vol = model vol, weekly hedge | 0.0103 | 0.9793 | 0.1061 | -2.7274 | 2.4184 | no |
| realised vol 30%, daily hedge | -3.9405 | 1.8925 | 0.2051 | -8.7199 | -0.8276 | no |

Premium is 9.2270. Limit: standard deviation of error within 10% of premium.

## 6. Ongoing Monitoring

Metrics, thresholds and current status. Thresholds are policy choices set for this exercise; Green is at or below the green bound, Amber at or below the amber bound, Red above it.

| metric | value | green_max | amber_max | status |
|---|---|---|---|---|
| max_abs_z_vs_mc | 1.99785 | 3 | 4 | Green |
| max_abs_diff_european_vs_fd | 0.000386461 | 0.01 | 0.03 | Green |
| max_abs_diff_american_tree_vs_fd | 0.00253156 | 0.01 | 0.03 | Green |
| max_parity_residual | 1.78e-14 | 1e-09 | 1e-06 | Green |
| max_rel_diff_delta_european | 0.000915138 | 0.01 | 0.03 | Green |
| tree_oscillation_amplitude | 0.0006146 | 0.005 | 0.02 | Green |
| rmse_vol_pts_in_band | 1.16052 | 0.5 | 1 | Red |
| rmse_vol_pts_all_quotes | 3.73494 | 1 | 2 | Red |
| hedge_std_over_premium_model_vol | 0.0478123 | 0.05 | 0.1 | Green |
| hedge_std_over_premium_vol_shock | 0.205102 | 0.1 | 0.2 | Red |

Statuses: 7 Green, 0 Amber, 3 Red. Re-run the pack on every model change and at least quarterly.

## 7. Limitations and Assumptions

- The market data in section 5 is synthetic; the smile is a stated function, not an observation.
- Monte Carlo results carry sampling error; the challenger Greeks use common random numbers and finite differences.
- The PDE challenger covers European options and American puts. American calls and discrete dividends are untested.
- Rates and dividend yield are constant; no stochastic volatility, jumps or funding effects are modelled or tested.
- Monitoring thresholds are exercise policy, not regulatory values.

## 8. Findings and Recommendations

Severity scale: High (limit breached with material error, or a conceptual check fails), Medium (limit breached or untested scope that affects use), Low (observation with small measured impact), Info (test passed, recorded).

| ID | Severity | Area | Evidence | Recommendation |
|---|---|---|---|---|
| F1 | High | Conceptual soundness / outcomes | Flat volatility misprices a skewed market: max implied-vol error 7.53 vol points, RMSE 3.73 points over 20 quotes; 4 of 8 in-band quotes breach the 1.0 point limit. | Restrict use to the moneyness band where the limit holds, or replace the flat surface with a strike and maturity dependent volatility input. |
| F2 | Info | Developmental evidence | European closed form agrees with Monte Carlo (max |z| 2.00, 38 of 40 95% intervals contain it) and with the PDE (max difference 0.0004). | None. Keep the benchmark in the monitoring pack. |
| F3 | Info | Developmental evidence | American put tree (500 steps) versus PDE challenger: max difference 0.0025 (versus 5000-step tree: 0.0023). | Raise tree steps or add Richardson extrapolation if the tolerance is tightened. |
| F4 | Info | Developmental evidence | Binomial price oscillates with step parity: spread 0.00061 across 200 to 203 steps; measured convergence order 1.00 (about 1 expected). | Use an odd-even smoothing (for example Leisen-Reimer or averaging two step counts). |
| F5 | Info | Developmental evidence | European delta agrees with the Monte Carlo challenger within 0.09%; gamma differences up to 2.4% reflect challenger sampling noise, not model error. | Treat the analytic Greeks as the reference; validate the challenger's gamma with a larger sample. |
| F6 | Low | Outcomes analysis | Daily delta hedging error is 4.8% of premium at the model volatility and 20.5% when realised volatility is 30% against the model's flat 20%; mean error -3.941. | Add a volatility-uncertainty reserve or monitor realised against implied volatility. |
| F7 | Medium | Limitations and assumptions | Discrete dividends are not supported: the model takes a continuous yield, which misstates early exercise value for American options around ex-dividend dates. No numeric test was possible. | Add a discrete dividend treatment or limit the model to indices and low-dividend underlyings. |
| F8 | Low | Limitations and assumptions | American calls priced by the front-office tree have no independent challenger: the PDE solver implemented here supports American puts only. | Extend the challenger to calls before approving the model for American calls with dividends. |
| F9 | Info | Conceptual soundness | 12 of 12 conceptual soundness checks passed. | None. |

## 9. Governance and Use Conditions

Section VI expects clear roles and accountability, managed conflicts of interest, documentation and tracking of recommendations and exceptions. For this exercise:

- Roles: the pricer, the challengers and this report were all written by one author, so independence between development and validation is simulated and not demonstrated. In practice this report would sit with a separate validation team with standing to require changes (the effective challenge described in section III).
- Documentation and tracking: every finding has an ID, severity and recommendation in section 8, and the table is written to `examples/output/validation_findings.csv` for status tracking.
- Use before validation: none. No exception is recorded.
- Extension of use (section IV): until a smile-aware model replaces it, pricing outside the in-band moneyness range should be limited or overlaid, and any new use should trigger its own review.

## 10. Conclusion

Not approved: at least one High finding is open.

Open High findings: F1. Open Medium findings: F7.
