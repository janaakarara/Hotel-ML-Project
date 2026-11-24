# Stage 3: Column Removal Decisions

## 1. Objective

Stage 3 removes the 11 columns selected for removal based on the Stage 2 redundancy analysis.
The purpose is to reduce redundant and low-information features before clustering.

## 2. Input and Output

- **Input:** `data/processed/h2_clustering_stage1.csv`
- **Output:** `data/processed/h2_clustering_stage2.csv`
- **Rows:** 53036
- **Original columns:** 22
- **Dropped columns:** 11
- **Retained columns:** 11

## 3. Reasons for Dropping Each Column

| Column | Reason | Evidence | Stage 2 reference |
|---|---|---|---|
| `Babies` | Near-constant | 99.31% of rows contain 0. | `stage2a_constancy.png` |
| `DaysInWaitingList` | Near-constant | 98.65% of rows contain 0. | `stage2a_constancy.png` |
| `PreviousCancellations` | Near-constant | 97.79% of rows contain 0. | `stage2a_constancy.png` |
| `PreviousBookingsNotCanceled` | Near-constant | 97.14% of rows contain 0. | `stage2a_constancy.png` |
| `IsRepeatedGuest` | Near-constant | 97.09% of rows contain 0. | `stage2a_constancy.png` |
| `DistributionChannel` | Redundancy | Redundant with MarketSegment (Cramer's V = 0.77), Agent (V = 0.86), and has_company (V = 0.82). | `stage2a_cramers_v.png` |
| `Agent` | Redundancy | Redundant with DistributionChannel (Cramer's V = 0.86). It also has 224 categories. | `stage2a_cramers_v.png` |
| `has_company` | Redundancy | Redundant with MarketSegment (Cramer's V = 0.85). Additionally, 94.83% of rows contain 0. | `stage2a_cramers_v.png` |
| `ArrivalDateWeekNumber` | Redundancy | Almost completely explained by ArrivalDateMonth (eta = 0.995). | `stage2a_eta.png` |
| `ArrivalDateDayOfMonth` | Judgment-based removal | The day is spread relatively evenly throughout the month and showed no meaningful relationship with other columns. | `stage2a_spearman.png` |
| `ArrivalDateYear` | Judgment-based removal | The dataset covers only July 2015 to August 2017. Year was treated as a calendar artifact rather than a useful clustering feature. | `Dataset date coverage` |

## 4. Retained Columns

- `LeadTime`
- `StaysInWeekendNights`
- `StaysInWeekNights`
- `Adults`
- `Children`
- `MarketSegment`
- `Meal`
- `Country`
- `CustomerType`
- `ReservedRoomType`
- `ArrivalDateMonth`

## 5. Final Result

Stage 3 reduced the dataset from **22 to 11 columns**, while preserving all **53036 rows** and their original order.

No encoding, scaling, grouping, or additional statistical analysis is performed in this stage.

## 6. Methodological Note

The removal decisions are based on the Stage 2 analysis and the documented judgment calls above. Stage 3 only applies these decisions; it does not recalculate the Stage 2 statistics.
