# Clustering stage 1: outcome and future columns

**Dropped: 9**  |  **Kept for the graph stage: 22**  |  Total: 31

## Dropped

| Column | Type | Reason |
|---|---|---|
| `IsCanceled` | outcome | The classification target. Clusters must be found without it. |
| `ReservationStatus` | outcome | Final status of the booking: a copy of IsCanceled. |
| `ReservationStatusDate` | outcome | Date the final status was set. Minus arrival date it predicts cancellation perfectly. |
| `AssignedRoomType` | future event | Paper: assigned for hotel operation reasons, near check-in. |
| `BookingChanges` | future event | Paper: changes counted 'until the moment of check-in or cancellation'. |
| `DepositType` | future event | Paper: computed from payments 'before arrival or cancellation date'. |
| `RequiredCarParkingSpaces` | suspected future | Not documented as future, but 0 of 1,888 bookings with parking were canceled. |
| `TotalOfSpecialRequests` | suspected future | Timing unclear; kept bookings show more requests at the same lead time. |
| `ADR` | outcome-like | Regression target; paper defines it from lodging transactions. Use it only to describe clusters. |

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
