import pandas as pd
import numpy as np
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer


print("=" * 70)
print("STEP 12 - TF-IDF FEATURE EXTRACTION")
print("=" * 70)


# ==========================================================
# 1. LOAD FEATURE ENGINEERED DATA
# ==========================================================

input_file = "output/featured_news.csv"

df = pd.read_csv(input_file)


print("\nInput Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ==========================================================
# 2. CHECK REQUIRED COLUMNS
# ==========================================================

if "Title" not in df.columns:

    raise ValueError(
        "Title column not found in featured_news.csv"
    )

if "Category" not in df.columns:

    raise ValueError(
        "Category column not found."
    )


# ==========================================================
# 3. CLEAN TEXT
# ==========================================================

text_data = (
    df["Title"]
    .fillna("")
    .astype(str)
)


# ==========================================================
# 4. TF-IDF VECTORISER
# ==========================================================

print("\nCreating TF-IDF features...")


tfidf = TfidfVectorizer(

    max_features=1000,

    ngram_range=(1, 2),

    min_df=2,

    max_df=0.95,

    sublinear_tf=True,

    stop_words="english"

)


tfidf_matrix = tfidf.fit_transform(
    text_data
)


print(
    "\nTF-IDF Matrix Shape:"
)

print(
    tfidf_matrix.shape
)


# ==========================================================
# 5. CONVERT TO DATAFRAME
# ==========================================================

tfidf_array = (
    tfidf_matrix
    .toarray()
)


tfidf_columns = [

    f"TFIDF_{i}"

    for i in range(
        tfidf_array.shape[1]
    )

]


tfidf_df = pd.DataFrame(

    tfidf_array,

    columns=tfidf_columns

)


# ==========================================================
# 6. ADD CATEGORY
# ==========================================================

tfidf_df["Category"] = (
    df["Category"].values
)


# ==========================================================
# 7. SAVE TF-IDF
# ==========================================================

output_file = (
    "output/news_tfidf.csv"
)


tfidf_df.to_csv(

    output_file,

    index=False

)


# ==========================================================
# 8. SAVE VECTORIZER
# ==========================================================

vectorizer_file = (
    "output/tfidf_vectorizer.pkl"
)


with open(
    vectorizer_file,
    "wb"
) as file:

    pickle.dump(
        tfidf,
        file
    )


# ==========================================================
# 9. FINAL REPORT
# ==========================================================

print("\n" + "=" * 70)
print("TF-IDF COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nOriginal Records:",
    len(df)
)

print(
    "TF-IDF Features:",
    len(tfidf_columns)
)

print(
    "Final TF-IDF Shape:",
    tfidf_df.shape
)

print(
    "\nSaved:"
)

print(
    output_file
)

print(
    vectorizer_file
)

print("\n" + "=" * 70)