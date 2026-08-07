import pandas as pd
import numpy as np
import os


print("=" * 70)
print("STEP 13 - MERGING ALL FEATURES")
print("=" * 70)


# ==========================================================
# 1. FILES
# ==========================================================

featured_file = "output/featured_news.csv"

embedding_file = "output/news_embeddings.csv"

tfidf_file = "output/news_tfidf.csv"


# ==========================================================
# 2. LOAD DATA
# ==========================================================

featured_df = pd.read_csv(
    featured_file
)

embedding_df = pd.read_csv(
    embedding_file
)

tfidf_df = pd.read_csv(
    tfidf_file
)


print("\nFeatured Data Shape:")
print(featured_df.shape)

print("\nEmbedding Data Shape:")
print(embedding_df.shape)

print("\nTF-IDF Data Shape:")
print(tfidf_df.shape)


# ==========================================================
# 3. CHECK ROW COUNTS
# ==========================================================

if not (
    len(featured_df)
    == len(embedding_df)
    == len(tfidf_df)
):

    raise ValueError(
        "Row counts do not match between "
        "featured, embedding and TF-IDF files."
    )


# ==========================================================
# 4. REQUIRED 15 ENGINEERED FEATURES
# ==========================================================

engineered_features = [

    "Title_Length",

    "Word_Count",

    "Avg_Word_Length",

    "Digit_Count",

    "Special_Character_Count",

    "Question_Count",

    "URL_Length",

    "Business_Keyword_Count",

    "Technology_Keyword_Count",

    "Politics_Keyword_Count",

    "Health_Keyword_Count",

    "Energy_Keyword_Count",

    "Organization_Count",

    "Person_Count",

    "Location_Count"

]


# ==========================================================
# 5. CHECK ENGINEERED FEATURES
# ==========================================================

missing_features = [

    feature

    for feature in engineered_features

    if feature not in featured_df.columns

]


if missing_features:

    raise ValueError(
        "Missing engineered features: "
        + str(missing_features)
    )


# ==========================================================
# 6. SELECT ENGINEERED FEATURES
# ==========================================================

engineered_df = featured_df[
    engineered_features
].copy()


# ==========================================================
# 7. CLEAN NUMERIC VALUES
# ==========================================================

engineered_df = (
    engineered_df
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .fillna(0)
)


# ==========================================================
# 8. REMOVE CATEGORY FROM TF-IDF
# ==========================================================

if "Category" in tfidf_df.columns:

    tfidf_features_df = tfidf_df.drop(
        columns=["Category"]
    )

else:

    tfidf_features_df = tfidf_df.copy()


# ==========================================================
# 9. REMOVE NON-FEATURE COLUMNS FROM EMBEDDINGS
# ==========================================================

embedding_non_features = [

    "Category",
    "Title",
    "URL",
    "Group Leader"

]


embedding_features_df = (
    embedding_df.drop(
        columns=[
            column

            for column
            in embedding_non_features

            if column in embedding_df.columns
        ]
    )
)


# ==========================================================
# 10. KEEP ONLY NUMERIC EMBEDDING COLUMNS
# ==========================================================

embedding_features_df = (
    embedding_features_df
    .select_dtypes(
        include=[np.number]
    )
)


# ==========================================================
# 11. KEEP ONLY NUMERIC TF-IDF COLUMNS
# ==========================================================

tfidf_features_df = (
    tfidf_features_df
    .select_dtypes(
        include=[np.number]
    )
)


# ==========================================================
# 12. CONVERT TO NUMERIC
# ==========================================================

embedding_features_df = (
    embedding_features_df
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .fillna(0)
)


tfidf_features_df = (
    tfidf_features_df
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .fillna(0)
)


# ==========================================================
# 13. RESET INDEX
# ==========================================================

engineered_df.reset_index(
    drop=True,
    inplace=True
)

embedding_features_df.reset_index(
    drop=True,
    inplace=True
)

tfidf_features_df.reset_index(
    drop=True,
    inplace=True
)


# ==========================================================
# 14. MERGE FEATURES
# ==========================================================

print("\nMerging features...")


combined_features = pd.concat(

    [

        embedding_features_df,

        tfidf_features_df,

        engineered_df

    ],

    axis=1

)


# ==========================================================
# 15. CHECK DUPLICATE COLUMNS
# ==========================================================

duplicate_columns = (
    combined_features.columns[
        combined_features.columns.duplicated()
    ]
)


if len(duplicate_columns) > 0:

    print(
        "\nDuplicate columns found:"
    )

    print(
        duplicate_columns.tolist()
    )

    combined_features = (
        combined_features.loc[
            :,
            ~combined_features.columns.duplicated()
        ]
    )


# ==========================================================
# 16. ADD TARGET
# ==========================================================

combined_features["Category"] = (
    featured_df["Category"].values
)


# ==========================================================
# 17. ADD TITLE FOR REFERENCE
# ==========================================================

combined_features["Title"] = (
    featured_df["Title"].values
)


# ==========================================================
# 18. FINAL SHAPE
# ==========================================================

print("\n" + "=" * 70)
print("MERGED FEATURE INFORMATION")
print("=" * 70)


print(
    "\nEmbedding Features:",
    embedding_features_df.shape[1]
)


print(
    "TF-IDF Features:",
    tfidf_features_df.shape[1]
)


print(
    "Engineered Features:",
    engineered_df.shape[1]
)


print(
    "\nTotal Numeric Features:",
    (
        embedding_features_df.shape[1]
        + tfidf_features_df.shape[1]
        + engineered_df.shape[1]
    )
)


print(
    "\nFinal Dataset Shape:",
    combined_features.shape
)


# ==========================================================
# 19. CHECK MISSING VALUES
# ==========================================================

numeric_part = (
    combined_features
    .drop(
        columns=[
            "Category",
            "Title"
        ]
    )
)


missing_values = (
    numeric_part.isna()
    .sum()
    .sum()
)


print(
    "\nTotal Missing Numeric Values:",
    missing_values
)


# ==========================================================
# 20. CHECK INFINITE VALUES
# ==========================================================

infinite_values = (
    np.isinf(
        numeric_part
        .select_dtypes(
            include=[np.number]
        )
    )
    .sum()
    .sum()
)


print(
    "Total Infinite Values:",
    infinite_values
)


# ==========================================================
# 21. REPLACE INFINITE VALUES
# ==========================================================

combined_features = (
    combined_features
    .replace(
        [np.inf, -np.inf],
        0
    )
)


# ==========================================================
# 22. SAVE
# ==========================================================

output_file = (
    "output/combined_features.csv"
)


combined_features.to_csv(

    output_file,

    index=False

)


# ==========================================================
# 23. FINAL REPORT
# ==========================================================

print("\n" + "=" * 70)
print("FEATURE MERGING COMPLETED SUCCESSFULLY")
print("=" * 70)


print(
    "\nSaved File:"
)

print(
    output_file
)


print(
    "\nRows:",
    combined_features.shape[0]
)


print(
    "Columns:",
    combined_features.shape[1]
)


print(
    "\n15 Engineered Features Included: YES"
)

print(
    "TF-IDF Included: YES"
)

print(
    "Embeddings Included: YES"
)

print("\n" + "=" * 70)
print("STEP 13 COMPLETED")
print("=" * 70)