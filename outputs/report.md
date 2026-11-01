# H2 Preprocessing Report

## Pipeline

The preprocessing pipeline uses `sklearn.pipeline.Pipeline` and three named transformers implemented in `pipeline/preprocess.py`.

## Results

| Metric | Before | After |
|---|---:|---:|
| Rows | 53,239 | 53,235 |
| Columns | 31 | 28 |

### Applied transformers

1. `drop_columns`: dropped `ReservationStatus`, `ReservationStatusDate`, and `RequiredCarParkingSpaces`.
2. `drop_undefined_distribution_channel`: removed 4 rows where `DistributionChannel` equals `Undefined`.
3. `binarize_booking_changes`: changed `BookingChanges` so `0` remains `0` and every non-zero value becomes `1`.

### BookingChanges distribution

| Value | Count |
|---|---:|
| 0 | 44,494 |
| 1 | 8,741 |

## Verification

- Dropped columns absent: `True`
- `Undefined` distribution-channel rows absent: `True`
- `BookingChanges` contains only 0/1: `True`

## Output

The sklearn pipeline returns the transformed result as an in-memory pandas DataFrame. No processed CSV file is created.
