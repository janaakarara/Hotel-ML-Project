import os
import pandas as pd

os.makedirs("outputs", exist_ok=True)

# 1. Load the CSV
df = pd.read_csv(r"data\raw\H2.csv")
print("Rows, columns:", df.shape)

# 2. Missing marker in this file is the text 'none' (also handle NULL and blanks)
df["Company"] = df["Company"].astype(str).str.strip()
df["Company"] = df["Company"].replace(["none", "NULL", "", "nan"], pd.NA)

# 3. Flag company bookings
df["is_company"] = df["Company"].notna()

print("\nCompany bookings vs not:")
company_counts = df["is_company"].value_counts()
print(company_counts)
company_counts.to_csv(os.path.join("outputs", "company_counts.csv"), header=["count"])

# 4. Cancellation rate by company booking
summary = df.groupby("is_company")["IsCanceled"].agg(["mean", "count"])
print("\nCancellation rate by company booking:")
print(summary)
summary.to_csv(os.path.join("outputs", "company_cancellation_summary.csv"))

# 5. Concentration of company IDs (real companies only)
print("\nNumber of distinct companies:", df["Company"].nunique())
top_companies = df["Company"].value_counts().head(10)
print(top_companies)
top_companies.to_csv(os.path.join("outputs", "top_companies.csv"), header=["count"])

print("\nSaved to the outputs folder.")