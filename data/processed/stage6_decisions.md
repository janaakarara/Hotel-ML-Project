# Stage 6: split and rebalance

`HasChildren` is removed from the features and used to split the bookings. Each subset is clustered on its own.

- Weighting: `equal`

| Subset | Rows | Share |
|---|---|---|
| nonfamily | 48110 | 91.1% |
| family | 4707 | 8.9% |

Variance share per variable after this stage is in `stage6_weights.json` (with `equal` weighting every variable sits near the same share).
