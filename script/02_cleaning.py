import pandas as pd
import re
import os

print("=" * 70)
print("STEP 2 - DATA CLEANING")
print("=" * 70)

df = pd.read_excel("data/news_classification.xlsx")

print("Original Shape:", df.shape)

# Remove missing target
df = df.dropna(subset=["Category"])

# Remove missing title
df["Title"] = df["Title"].fillna("").astype(str)

df = df[df["Title"].str.strip() != ""]

# Clean category
df["Category"] = df["Category"].astype(str).str.strip()

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["Title_Clean"] = df["Title"].apply(clean_text)

# Remove empty cleaned titles
df = df[df["Title_Clean"] != ""]

# Remove duplicates
df = df.drop_duplicates(subset=["Title_Clean"])

df = df.reset_index(drop=True)

os.makedirs("output", exist_ok=True)

df.to_csv(
    "output/cleaned_news.csv",
    index=False,
    encoding="utf-8"
)

print("\nFinal Shape:", df.shape)

print("\nCategory Distribution:")
print(df["Category"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())

print("\n✅ Saved: output/cleaned_news.csv")