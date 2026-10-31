# Agent vs. Cancellation Analysis

- **Data file:** `data\raw\H2.csv`
- **Rows analyzed:** 53,239
- **Agents (including UNKNOWN):** 224
- **Overall cancellation rate:** 30.06%

## Statistical relationship

- Chi-square statistic: **4,206.33**
- Degrees of freedom: **223**
- P-value: **0.000e+00**
- Bias-corrected Cramer's V: **0.2735**
- Minimum expected cell count: **0.30**
- Cells with expected count below 5: **266** of 448

The chi-square test evaluates whether cancellation and agent are independent. Cramer's V describes the strength of the association; 0 means no association and larger values indicate stronger association.

> With this dataset's large sample size, a very small p-value is not by itself proof 
> of a practically large effect. Interpret the cancellation rates and Cramer's V together.

## Top 10 agents by cancellation rate

Agents are shown only if they have at least 30 bookings, to avoid overemphasizing very small samples.

| Agent | Canceled | Not canceled | Total | Cancellation rate | Rate difference (pp) |
|---|---:|---:|---:|---:|---:|
| 58 | 35 | 13 | 48 | 72.92% | +42.86 |
| 118 | 26 | 38 | 64 | 40.62% | +10.56 |
| 9 | 11496 | 17193 | 28689 | 40.07% | +10.01 |
| 229 | 84 | 131 | 215 | 39.07% | +9.01 |
| 29 | 30 | 49 | 79 | 37.97% | +7.91 |
| 1 | 455 | 767 | 1222 | 37.23% | +7.17 |
| 56 | 16 | 33 | 49 | 32.65% | +2.59 |
| 134 | 26 | 54 | 80 | 32.50% | +2.44 |
| 44 | 10 | 21 | 31 | 32.26% | +2.20 |
| 30 | 34 | 72 | 106 | 32.08% | +2.01 |

## Top 10 agents by booking volume

| Agent | Canceled | Not canceled | Total | Share of bookings | Cancellation rate | Rate difference (pp) |
|---|---:|---:|---:|---:|---:|---:|
| 9 | 11496 | 17193 | 28689 | 53.89% | 40.07% | +10.01 |
| none | 821 | 4718 | 5539 | 10.40% | 14.82% | -15.24 |
| 14 | 582 | 2759 | 3341 | 6.28% | 17.42% | -12.64 |
| 7 | 436 | 2858 | 3294 | 6.19% | 13.24% | -16.82 |
| 28 | 87 | 1406 | 1493 | 2.80% | 5.83% | -24.23 |
| 1 | 455 | 767 | 1222 | 2.30% | 37.23% | +7.17 |
| 8 | 347 | 769 | 1116 | 2.10% | 31.09% | +1.03 |
| 83 | 157 | 457 | 614 | 1.15% | 25.57% | -4.49 |
| 6 | 125 | 415 | 540 | 1.01% | 23.15% | -6.91 |
| 85 | 98 | 425 | 523 | 0.98% | 18.74% | -11.32 |

## Artifact files

- Full agent summary: [`agent_cancellation_summary.csv`](./agent_cancellation_summary.csv)
- Cancellation-rate chart: [`agent_cancellation_rates.png`](./agent_cancellation_rates.png)
