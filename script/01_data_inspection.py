import pandas as pd
import os

print("=" * 70)
print("STEP 1 - DATA INSPECTION")
print("=" * 70)

file_path = "data/news_classification.xlsx"

if not os.path.exists(file_path):
    print("❌ File not found:", file_path)
    exit()

df = pd.read_excel(file_path)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nCategory Distribution:")
print(df["Category"].value_counts(dropna=False))

print("\nUnique Categories:")
print(df["Category"].dropna().unique())

print("\nNumber of Classes:")
print(df["Category"].nunique())

print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)