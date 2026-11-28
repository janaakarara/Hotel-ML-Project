# Preprocessing
1. Drop "ReservationStatus" "ReservationStatusDate" "RequiredCarParkingSpaces"
2. Drop "Undefined" rows in "DistributionChannel"
3. Binarize "BookingChanges" Valu

# Prediction app
Run `streamlit run app.py` from the project root. Upload a booking CSV and
select the optimized Random Forest or MLP model. The uploaded CSV must contain
the booking columns used during training; `IsCanceled` is optional. Predictions
are shown first, and the model's held-out metrics are available from the
"See the metrics of this model" button. If `IsCanceled` is included, the app
also calculates metrics for the uploaded rows.