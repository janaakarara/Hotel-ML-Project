# Clustering stage 1: zero-adult rows, outcome and future columns

## Rows removed

**219 rows removed** (53036 -> 52817): bookings with `Adults = 0`. All of them have children but no adult, so they are most likely data errors and not a meaningful customer segment. This is done for clustering only; `h2_clean.csv` is unchanged.

**Dropped columns: 9**  |  **Kept for the graph stage: 22**  |  Total: 31

## Dropped

| Column | Type | Reason |
|---|---|---|
| `IsCanceled` | outcome | The classification target. Clusters must be found without it. |
| `ReservationStatus` | outcome | Final status of the booking: a copy of IsCanceled. |
| `ReservationStatusDate` | outcome | Date the final status was set. Minus arrival date it predicts cancellation perfectly. |
| `AssignedRoomType` | future event | Paper: assigned for hotel operation reasons, near check-in. |
| `BookingChanges` | future event | Paper: changes counted 'until the moment of check-in or cancellation'. |
| `DepositType` | future event | Paper: computed from payments 'before arrival or cancellation date'. |
| `RequiredCarParkingSpaces` | suspected future | Not documented as future, but 0 of 1,886 bookings with parking were canceled. |
| `TotalOfSpecialRequests` | suspected future | Timing unclear; kept bookings show more requests at the same lead time. |
| `ADR` | outcome-like | Regression target; paper defines it from lodging transactions. Use it only to describe clusters. |

The dropped columns are saved for the kept rows in `h2_clustering_profile_cols.csv` (same row order) and are used only to describe the clusters afterwards.

## Kept for now

These are not outcomes or future values. Redundancy, skew and grouping are decided from the plots.

| Column |
|---|
| `LeadTime` |
| `ArrivalDateYear` |
| `ArrivalDateMonth` |
| `ArrivalDateWeekNumber` |
| `ArrivalDateDayOfMonth` |
| `StaysInWeekendNights` |
| `StaysInWeekNights` |
| `Adults` |
| `Children` |
| `Babies` |
| `Meal` |
| `Country` |
| `MarketSegment` |
| `DistributionChannel` |
| `IsRepeatedGuest` |
| `PreviousCancellations` |
| `PreviousBookingsNotCanceled` |
| `ReservedRoomType` |
| `Agent` |
| `has_company` |
| `DaysInWaitingList` |
| `CustomerType` |
