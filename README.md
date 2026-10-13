# frtb-sa-validation-pack

A small, readable Python implementation of Basel FRTB market risk capital (standardised approach and a simplified internal models approach) on a synthetic trading book, plus an SR 11-7 style independent validation of an option pricer. Runtime dependencies are numpy and pandas only. Every input is synthetic and seeded.

## What is in it

| Package | Content |
|---|---|
| `frtb_val.pricing`, `market`, `instruments`, `portfolio` | Black-Scholes-Merton and CRR binomial pricers, a 46 instrument book (IRS, bonds, CDS, equities, FX forwards and options, equity options, exotics), bump-and-revalue sensitivities and curvature positions |
| `frtb_val.frtb_sa` | Sensitivities-based method (delta, vega, curvature for GIRR, CSR non-securitisation, equity, FX) with the three correlation scenarios and max selection, default risk charge, residual risk add-on, capital decomposition |
| `frtb_val.ima` | Expected shortfall at 97.5% with the liquidity horizon cascade, reduced set and stress period selection, ES_R,S x max(ES_F,C / ES_R,C, 1), IMCC, risk factor eligibility test, NMRF stress scenario charge, VaR backtesting with traffic lights and a binomial test |
| `frtb_val.pla` | P&L attribution test (Spearman, Kolmogorov-Smirnov), zones and desk eligibility |
| `frtb_val.validation` | Validation of a flat-volatility BSM plus American binomial pricer against Monte Carlo (antithetic, regression control variate) and Crank-Nicolson finite difference challengers; conceptual checks, benchmark grid, convergence, Greeks, outcomes analysis, findings, monitoring metrics, report writer |

## Run it

```
pip install numpy pandas
python -m unittest discover -s tests
python examples/run_sa.py
python examples/run_ima.py
python examples/run_pla.py
python examples/run_validation.py      # writes docs/validation_report.md
```

Examples write CSV and Markdown tables to `examples/output/`. The validation example takes about one minute.

## Results from the shipped runs (synthetic data)

Standardised approach, 46 instrument book (`examples/run_sa.py`):

| Component | Capital (USD) |
|---|---|
| SBM, low / medium / high correlation scenario | 26,759,075 / 25,796,822 / 24,718,432 |
| SBM selected (low binds) | 26,759,075 |
| DRC | 2,542,470 |
| RRAO | 552,035 |
| Total SA | 29,853,580 |

Internal models (`examples/run_ima.py`): 78 risk factors, 9 fail the eligibility test. ES_R,C / ES_F,C is 0.978. Stressed ES_R,S is 12.92 million, combined ES 13.20 million, IMCC 16.57 million, SES 1.90 million, IMCC plus SES 18.47 million. With all horizons forced to 10 days the current ES is 3.50 million against 3.66 million with the cascade. Backtesting, 250 observations at 99%: the model VaR shows 5 exceptions (amber), a VaR scaled to 70% shows 10 (red).

P&L attribution by desk (`examples/run_pla.py`): RATES green (Spearman 0.975, KS 0.040), FX green, EQD amber (Spearman 0.796), CREDIT red (Spearman 0.581, standardised approach).

Validation (`examples/run_validation.py`): the closed form agrees with Monte Carlo (max absolute z-score 2.00 over 40 comparisons) and finite differences (max difference 0.0004). The 500 step American put tree is within 0.0025 of the PDE challenger. The flat 20% volatility fails against a stated smile: implied-vol errors up to 7.5 vol points, 4 of 8 in-band quotes breach the 1 point limit. Findings: 1 High, 1 Medium, 2 Low, 5 Info. Conclusion: not approved while the High finding is open. Monitoring: 7 Green, 0 Amber, 3 Red. The full report is `docs/validation_report.md`.

## Tests

190 unittest tests (SBM 38, DRC and portfolio 43, IMA and PLA 64, validation 45), all passing. They include an analytic two-factor SBM case, charge reduction under hedge offset, correlation scenario max selection, zero curvature for a linear book, ES monotone in confidence, equal liquidity horizons reducing to plain ES, KS and Spearman against known values, backtesting zone probabilities, put-call parity, binomial convergence to Black-Scholes and Monte Carlo within its confidence interval.

## Limits and status of parameters

- Risk weights, correlations, gammas, DRC weights, the RFET criteria, the ES and SES formulas, backtesting zones and PLA thresholds follow the Basel text (BCBS d457, January 2019, MAR20 to MAR33), with the paragraph cited next to each constant in `frtb_sa/params.py` and the `ima` modules. They were read from the published document, not from memory. Jurisdictional rules (for example the US or EU implementations) differ and are not modelled.
- Stylized, not from the text: the synthetic book and market, the factor model and Student-t panel behind IMA P&L, delta-gamma P&L, the planted stress window, the mapping of the synthetic risk factors to liquidity horizon categories, the reduced set selection rule, the NMRF stress scenario (a simplified calibration), desk definitions for PLA, and all validation limits, monitoring thresholds and severity scales.
- The SR 11-7 report follows the guidance's headings and is a paraphrase of structure, not a compliance artefact. The synthetic smile and the findings are illustrative.
- The book contains no GIRR or CSR options, so GIRR vega, CSR vega and CSR curvature are implemented and unit tested on constructed inputs but show zero in the shipped run.
- Not implemented: securitisation and correlation trading portfolio treatments, the commodity risk class, DRC under the IMA, and jurisdiction specific add-ons. An LLM based regulatory assistant is future work only.
- Monte Carlo and hedging outputs depend on the seeds and path counts set in each example's config block.

## Layout

```
src/frtb_val/   package
tests/          unittest suites
examples/       runnable scripts, outputs in examples/output/
docs/           methodology.md, validation_report.md (generated)
```
