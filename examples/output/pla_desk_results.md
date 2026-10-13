# P&L attribution test by desk (synthetic)

| desk | spearman | ks | ks_p_value | pla_zone | exceptions_99 | exceptions_975 | outcome |
|---|---|---|---|---|---|---|---|
| RATES | 0.9754 | 0.0400 | 0.9883 | green | 3 | 10 | IMA eligible |
| FX | 0.9404 | 0.0560 | 0.8280 | green | 2 | 7 | IMA eligible |
| EQD | 0.7957 | 0.0640 | 0.6852 | amber | 0 | 4 | IMA with capital surcharge (PLA amber) |
| CREDIT | 0.5808 | 0.0480 | 0.9356 | red | 3 | 6 | STANDARDISED (PLA red) |
