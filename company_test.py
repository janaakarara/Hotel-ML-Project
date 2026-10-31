import pandas as pd

# 1. Load the CSV
df = pd.read_csv("Hotel_Data.csv")
print("Rows, columns:", df.shape)

# 2. Missing marker in this file is the text 'none' (also handle NULL and blanks)
df["Company"] = df["Company"].astype(str).str.strip()
df["Company"] = df["Company"].replace(["none", "NULL", "", "nan"], pd.NA)

# 3. Flag company bookings
df["is_company"] = df["Company"].notna()

print("\nCompany bookings vs not:")
print(df["is_company"].value_counts())

# 4. Cancellation rate by company booking
summary = df.groupby("is_company")["IsCanceled"].agg(["mean", "count"])
print("\nCancellation rate by company booking:")
print(summary)

# 5. Concentration of company IDs (real companies only)
print("\nNumber of distinct companies:", df["Company"].nunique())
top_companies = df["Company"].value_counts().head(10)
print(top_companies)

