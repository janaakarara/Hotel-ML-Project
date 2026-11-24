# Stage 5: grouping and skew fixes

Applies the decisions made from the stage 4 figures. No row is removed and the row order is unchanged.

- **Input:** `data/processed/h2_clustering_stage3.csv` (53036 rows, 11 columns)
- **Output:** `h2_clustering_stage5.csv`

## 1. Country: top 5 countries plus continents

Country had **167 values**, which would create 167 encoded columns and swamp every other feature. It now has **11 groups**.

**Why not pure continents?** About 86% of bookings are European, so one huge Europe group would hide big differences. For example the median booking is made 26 days ahead for Portugal but 83 days ahead for Germany. So the 5 biggest countries stay separate and only the rest are grouped by continent.

| Group | % of rows |
|---|---|
| PRT | 24.4 |
| Rest of Europe | 23.7 |
| FRA | 13.9 |
| GBR | 8.5 |
| DEU | 8.4 |
| ESP | 7.1 |
| Asia | 5.1 |
| South America | 3.6 |
| North America | 2.9 |
| Africa | 1.6 |
| Other | 0.7 |

Notes: `CN` is China in this dataset and is mapped to Asia by hand. Oceania and junk codes (`Unknown`, `TMP`, `ATA` ...) fall into `Other`. Russia counts as Europe and Turkey and Cyprus as Asia.

## 2. Rare categories in the other columns

Rule: a value under 1% of rows is merged. Where a value has an obvious close relative it is merged with it, otherwise it goes into `Other`.

| Column | Merge | Why | Categories |
|---|---|---|---|
| Meal | FB -> HB/FB | FB is 0.02% of rows. Both FB and HB are meal plans beyond breakfast, so they are merged. | 4 -> 3 |
| CustomerType | Group -> Transient-Party/Group | Group is 0.5% of rows. Both are bookings tied to other bookings, so they are merged. | 4 -> 3 |
| MarketSegment | Complementary -> Other, Aviation -> Other | Both are under 1% of rows and are special cases with no close relative. | 7 -> 6 |
| ReservedRoomType | G -> Other, C -> Other | Both are under 1% of rows. Room codes are anonymous, so there is no meaning to merge by. | 7 -> 6 |

The merges of Meal and CustomerType are judgment calls. If you prefer, change them to `Other` in `MERGES`.

## 3. Skew fixes

Skew under 0.5 is balanced, 0.5 to 1 moderate, over 1 high. Long tails would stretch the distance between points and let a few extreme bookings dominate the clusters. For each column we used the method with the lowest skew in the stage 4 figure.

| Column | Method | New column | Skew before | Skew after |
|---|---|---|---|---|
| LeadTime | Yeo-Johnson | LeadTime_yj | 1.57 | -0.12 |
| StaysInWeekNights | Yeo-Johnson | StaysInWeekNights_yj | 2.52 | 0.02 |
| StaysInWeekendNights | Square root | StaysInWeekendNights_sqrt | 1.00 | -0.01 |

**Children** became a 0/1 flag (`HasChildren`). 91% of values are 0, so no transform can fix the skew (every method stayed above 2.8 in the stage 4 figure). The question that matters for segments is whether children are present.

**Adults** was checked and left unchanged (skew -0.24, already balanced).

**ArrivalDateMonth** is unchanged here. It is a cyclical category and is handled in the encoding stage.

## 4. Result

11 columns in, 11 columns out, 53036 rows kept.

Final columns: `Adults`, `MarketSegment`, `Meal`, `Country`, `CustomerType`, `ReservedRoomType`, `ArrivalDateMonth`, `LeadTime_yj`, `StaysInWeekNights_yj`, `StaysInWeekendNights_sqrt`, `HasChildren`

The skew fixes learn nothing from any target. Clustering is unsupervised, so fitting the Yeo-Johnson setting on all rows leaks nothing. The fitted settings are saved in `stage5_transform_params.json` so the same transform can be repeated on new bookings.

The untransformed columns stay in `h2_clustering_stage3.csv` (same row order) for describing the clusters afterwards.
