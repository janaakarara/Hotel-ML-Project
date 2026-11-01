# Country vs. Cancellation Analysis

- **Data file:** `data\raw\H2.csv`
- **Rows analyzed:** 53,239
- **Countries (including UNKNOWN):** 166
- **Overall cancellation rate:** 30.06%

## Statistical relationship

- Chi-square statistic: **1,958.85**
- Degrees of freedom: **165**
- P-value: **5.293e-304**
- Bias-corrected Cramer's V: **0.1836**
- Minimum expected cell count: **0.30**
- Cells with expected count below 5: **167** of 332

The chi-square test evaluates whether cancellation and country are independent. Cramer's V describes the strength of the association; 0 means no association and larger values indicate stronger association.

> With this dataset's large sample size, a very small p-value is not by itself proof 
> of a practically large effect. Interpret the cancellation rates and Cramer's V together.

## Top 10 countries by cancellation rate

Countries are shown only if they have at least 30 bookings, to avoid overemphasizing very small samples.

| Country | Canceled | Not canceled | Total | Cancellation rate | Rate difference (pp) |
|---|---:|---:|---:|---:|---:|
| ARE | 32 | 4 | 36 | 88.89% | +58.83 |
| SAU | 30 | 14 | 44 | 68.18% | +38.12 |
| AGO | 170 | 132 | 302 | 56.29% | +26.23 |
| TUN | 17 | 19 | 36 | 47.22% | +17.16 |
| TUR | 90 | 103 | 193 | 46.63% | +16.57 |
| THA | 16 | 19 | 35 | 45.71% | +15.65 |
| MAR | 72 | 89 | 161 | 44.72% | +14.66 |
| UKR | 20 | 25 | 45 | 44.44% | +14.38 |
| CHN | 530 | 675 | 1205 | 43.98% | +13.92 |
| RUS | 166 | 220 | 386 | 43.01% | +12.94 |

## Top 10 countries by booking volume

| Country | Canceled | Not canceled | Total | Share of bookings | Cancellation rate | Rate difference (pp) |
|---|---:|---:|---:|---:|---:|---:|
| PRT | 4869 | 8245 | 13114 | 24.63% | 37.13% | +7.07 |
| FRA | 1544 | 5854 | 7398 | 13.90% | 20.87% | -9.19 |
| GBR | 1297 | 3221 | 4518 | 8.49% | 28.71% | -1.35 |
| DEU | 931 | 3534 | 4465 | 8.39% | 20.85% | -9.21 |
| ESP | 1118 | 2659 | 3777 | 7.09% | 29.60% | -0.46 |
| ITA | 1003 | 1639 | 2642 | 4.96% | 37.96% | +7.90 |
| BEL | 367 | 1323 | 1690 | 3.17% | 21.72% | -8.34 |
| BRA | 635 | 951 | 1586 | 2.98% | 40.04% | +9.98 |
| NLD | 299 | 1131 | 1430 | 2.69% | 20.91% | -9.15 |
| USA | 394 | 1019 | 1413 | 2.65% | 27.88% | -2.18 |

## Artifact files

- Full country summary: [`country_cancellation_summary.csv`](./country_cancellation_summary.csv)
- Cancellation-rate chart: [`country_cancellation_rates.png`](./country_cancellation_rates.png)
