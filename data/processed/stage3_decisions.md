# Stage 3b: grouping and skew fixes

Applies the decisions made from the stage 3a figures. No row is removed and the row order is unchanged.

- **Input:** `data/processed/h2_clustering_stage2.csv` (52817 rows, 9 columns)
- **Output:** `h2_clustering_stage3.csv`

## 1. Country: 4 groups

Country had **167 values**, which would create 167 encoded columns and swamp every other feature. It now has **4 groups**, each at least 14% of rows.

| Group | Members |
|---|---|
| Portugal | PRT (the hotel is in Lisbon: domestic guests) |
| Latin Europe | FRA, ESP, ITA (Portugal's nearest Latin-language neighbours) |
| Rest of Europe | all other European codes (GBR, DEU, BEL, NLD, CHE, IRL, AUT, Nordics, Eastern Europe ...) |
| Outside Europe | Asia, the Americas, Africa, Oceania, and unknown / junk codes (`Unknown`, `TMP`, `ATA` ...) |

**Why this grouping?** The earlier version (top 5 countries + continents) gave 11 columns and let Country outvote the numeric features. Four groups of 14% to 36% remove every rare category and need only 4 columns. Groups were chosen for balance and interpretability, not strict geography: 'Latin Europe' is only the three Latin-language neighbours, and 'Rest of Europe' is everyone else. Germany and Britain are merged, which averages two different booking habits; lead time is already a clustering feature, so most of that difference is still captured.

| Group | % of rows |
|---|---|
| Portugal | 24.4 |
| Latin Europe | 26.1 |
| Rest of Europe | 35.6 |
| Outside Europe | 13.9 |

**Do the groups differ?** (medians from the untransformed stage 2 columns)

| Group | Median lead time (days) | Median week nights | Median weekend nights | Share with children |
|---|---|---|---|---|
| Portugal | 26 | 2 | 0 | 6.6% |
| Latin Europe | 49 | 2 | 1 | 10.7% |
| Rest of Europe | 74 | 2 | 1 | 8.5% |
| Outside Europe | 40 | 2 | 1 | 10.6% |

Lead time is the clear separator; stay length and the children share differ little. Note: Russia counts as Europe, Turkey and Cyprus as Asia (Outside Europe), `CN` is China.

## 2. Rare categories in the other columns

Rule: a value under 1% of rows is merged. Where a value has an obvious close relative it is merged with it, otherwise it goes into `Other`.

| Column | Merge | Why | Categories |
|---|---|---|---|
| CustomerType | Group -> Transient-Party/Group | Group is 0.5% of rows. Both are bookings tied to other bookings, so they are merged. | 4 -> 3 |
| MarketSegment | Complementary -> Other, Aviation -> Other | Both are under 1% of rows and are special cases with no close relative. | 7 -> 6 |

The CustomerType merge is a judgment call. If you prefer, change them to `Other` in `MERGES`.

## 3. Skew fixes

Skew under 0.5 is balanced, 0.5 to 1 moderate, over 1 high. Long tails would stretch the distance between points and let a few extreme bookings dominate the clusters. For each column we used the method with the lowest skew in the stage 3a figure.

| Column | Method | New column | Skew before | Skew after |
|---|---|---|---|---|
| LeadTime | Yeo-Johnson | LeadTime_yj | 1.58 | -0.12 |
| StaysInWeekNights | Yeo-Johnson | StaysInWeekNights_yj | 2.52 | 0.02 |
| StaysInWeekendNights | Square root | StaysInWeekendNights_sqrt | 1.00 | -0.01 |

**Children** became a 0/1 flag (`HasChildren`). 91% of values are 0, so no transform can fix the skew (every method stayed above 2.8 in the stage 3a figure). The question that matters for segments is whether children are present.

**Adults** was checked and left unchanged (skew -0.24, already balanced).

**ArrivalDateMonth** is unchanged here. It is a cyclical category and is handled in the encoding stage.

## 4. Result

9 columns in, 9 columns out, 52817 rows kept.

Final columns: `Adults`, `MarketSegment`, `Country`, `CustomerType`, `ArrivalDateMonth`, `LeadTime_yj`, `StaysInWeekNights_yj`, `StaysInWeekendNights_sqrt`, `HasChildren`

The skew fixes learn nothing from any target. Clustering is unsupervised, so fitting the Yeo-Johnson setting on all rows leaks nothing. The fitted settings are saved in `stage3_transform_params.json` so the same transform can be repeated on new bookings.

The untransformed columns stay in `h2_clustering_stage2.csv` (same row order) for describing the clusters afterwards.
