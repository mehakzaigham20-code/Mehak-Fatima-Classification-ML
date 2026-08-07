import pandas as pd
import re

print("=" * 70)
print("STEP 3 - TEXT PREPROCESSING")
print("=" * 70)

df = pd.read_csv("output/cleaned_news.csv")

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["Processed_Title"] = df["Title_Clean"].apply(preprocess_text)

df["Title_Length"] = df["Processed_Title"].str.len()

df["Word_Count"] = df["Processed_Title"].str.split().str.len()

df.to_csv(
    "output/preprocessed_news.csv",
    index=False,
    encoding="utf-8"
)

print("\nShape:", df.shape)

print("\nNew Features:")
print([
    "Processed_Title",
    "Title_Length",
    "Word_Count"
])

print("\n✅ Saved: output/preprocessed_news.csv")