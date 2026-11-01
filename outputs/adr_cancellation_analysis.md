# ADR vs. Cancellation Analysis

- **Data file:** `data\raw\H2.csv`
- **Rows analyzed:** 53,239
- **Overall cancellation rate:** 30.06%

## Direct answer: ADR = 0

- Rows with ADR = 0: **952**
- Canceled among those rows: **96**
- Not canceled among those rows: **856**
- Cancellation rate for ADR = 0: **10.08%**

### No — ADR = 0 does not always imply IsCanceled = 1.

This is calculated directly from all observed rows. A zero ADR is associated with a smaller cancellation rate than ADR values that are not zero, but it does not determine cancellation.

## ADR = 0 vs ADR != 0

| ADR group | Canceled | Not canceled | Total | Cancellation rate |
|---|---:|---:|---:|---:|
| ADR = 0 | 96 | 856 | 952 | 10.08% |
| ADR != 0 | 15,908 | 36,379 | 52,287 | 30.42% |

## Numeric ADR relationship

- Point-biserial correlation with IsCanceled: **0.0922**
- P-value: **7.068e-101**

The point-biserial correlation measures the linear association between numeric ADR and the binary target.

## Cancellation rate by ADR band

| ADR band | Canceled | Not canceled | Total | Cancellation rate | Rate difference (pp) |
|---|---:|---:|---:|---:|---:|
| 0 | 96 | 856 | 952 | 10.08% | -19.98 |
| >0-50 | 116 | 428 | 544 | 21.32% | -8.74 |
| >50-100 | 6054 | 16779 | 22833 | 26.51% | -3.55 |
| >100-150 | 6900 | 13870 | 20770 | 33.22% | +3.16 |
| >150-200 | 2253 | 4249 | 6502 | 34.65% | +4.59 |
| >200-300 | 573 | 1023 | 1596 | 35.90% | +5.84 |
| >300-500 | 12 | 29 | 41 | 29.27% | -0.79 |
| >500 | 0 | 1 | 1 | 0.00% | -30.06 |

## Artifact files

- Full ADR-band summary: [`adr_cancellation_summary.csv`](./adr_cancellation_summary.csv)
- Comparison chart: [`adr_cancellation_rate_chart.png`](./adr_cancellation_rate_chart.png)
