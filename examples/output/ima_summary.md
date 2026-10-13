# IMA results (synthetic book and synthetic factor panel)

| item | value |
|---|---|
| risk factors total | 78.0000 |
| risk factors failing RFET (NMRF) | 9.0000 |
| reduced set size (modellable) | 30.0000 |
| ES_F,C current, full set (USD) | 3,655,733.8164 |
| ES_R,C current, reduced set (USD) | 3,575,821.2906 |
| ES_R,C / ES_F,C (needs >= 0.75) | 0.9781 |
| ES_R,S stressed, reduced set (USD) | 12,915,689.9990 |
| stressed window start scenario (planted at day 700) | 665.0000 |
| ES = ES_R,S x max(ES_F,C / ES_R,C, 1) (USD) | 13,204,330.0975 |
| IMCC with rho = 0.5 (USD) | 16,570,903.3422 |
| SES (USD) | 1,896,905.8460 |
| IMCC + SES (USD) | 18,467,809.1882 |
| plain 10-day ES, all horizons forced to 10 (USD) | 3,503,879.4167 |
| liquidity-adjusted cascade ES, current window (USD) | 3,655,733.8164 |
| amber zone starts at (250 obs, 99%) | 5.0000 |
| red zone starts at (250 obs, 99%) | 10.0000 |

ES by broad risk class (stressed, constrained):

| class | ES |
|---|---|
| CS | 2,878,573 |
| EQ | 4,077,171 |
| FX | 1,889,897 |
| IR | 11,091,835 |

Plain 10-day ES by confidence level:

| alpha | ES_10d_plain |
|---|---|
| 1 | 2,707,462 |
| 1 | 3,249,953 |
| 1 | 3,503,879 |
| 1 | 3,687,669 |

Backtesting, last 250 days:

| confidence | var | n | exceptions | expected | zone | p_too_many | kupiec_p |
|---|---|---|---|---|---|---|---|
| 0.9900 | model VaR | 250 | 5 | 2.5000 | amber | 0.1078 | 0.1619 |
| 0.9900 | VaR scaled to 70% | 250 | 10 | 2.5000 | red | 0.0003 | 0.0003 |
| 0.9750 | model VaR | 250 | 8 | 6.2500 | green | 0.2897 | 0.4965 |
| 0.9750 | VaR scaled to 70% | 250 | 15 | 6.2500 | amber | 0.0018 | 0.0026 |

Factors failing the RFET:

| factor_id | liquidity_horizon | n_obs | min_obs_in_90d | criterion_1 | criterion_2 |
|---|---|---|---|---|---|
| CSR|delta|4|IND_A|BOND|1.0 | 40 | 48 | 3 | no | no |
| CSR|delta|11|FIN_HY|BOND|1.0 | 60 | 23 | 0 | no | no |
| CSR|delta|11|FIN_HY|BOND|5.0 | 60 | 26 | 0 | no | no |
| CSR|delta|11|FIN_HY|CDS|0.5 | 60 | 21 | 2 | no | no |
| CSR|delta|11|FIN_HY|CDS|1.0 | 60 | 37 | 0 | no | no |
| CSR|delta|11|FIN_HY|CDS|5.0 | 60 | 23 | 3 | no | no |
| CSR|delta|13|CNS_HY|BOND|1.0 | 60 | 25 | 0 | no | no |
| CSR|delta|13|CNS_HY|BOND|3.0 | 60 | 29 | 2 | no | no |
| EQ|vega|10|EQ_SMALL|VOL|1.0 | 60 | 32 | 0 | no | no |
