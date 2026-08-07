import pandas as pd
import numpy as np
import os

from sklearn.decomposition import PCA


print("=" * 70)
print("STEP 07 - PCA ON NEWS EMBEDDINGS")
print("=" * 70)


# ==========================================================
# 1. LOAD EMBEDDINGS
# ==========================================================

input_file = "output/news_embeddings.csv"

df = pd.read_csv(input_file)

print("\nEmbedding Dataset Shape:")
print(df.shape)


# ==========================================================
# 2. SELECT EMBEDDING COLUMNS
# ==========================================================

embedding_columns = [
    column
    for column in df.columns
    if column.startswith("Embedding_")
]

if len(embedding_columns) == 0:
    raise ValueError(
        "No embedding columns found."
    )


print(
    "\nOriginal Embedding Dimensions:",
    len(embedding_columns)
)


X_embeddings = df[
    embedding_columns
].values


# ==========================================================
# 3. PCA
# ==========================================================

n_components = 50

print(
    "\nReducing embeddings:",
    len(embedding_columns),
    "->",
    n_components
)

pca = PCA(
    n_components=n_components,
    random_state=42
)

X_pca = pca.fit_transform(
    X_embeddings
)


# ==========================================================
# 4. EXPLAINED VARIANCE
# ==========================================================

explained_variance = (
    pca.explained_variance_ratio_
)

total_variance = (
    explained_variance.sum()
)

print(
    "\nTotal Explained Variance:",
    round(
        total_variance * 100,
        2
    ),
    "%"
)


# ==========================================================
# 5. CREATE PCA DATAFRAME
# ==========================================================

pca_columns = [
    f"PCA_{i+1}"
    for i in range(n_components)
]

pca_df = pd.DataFrame(
    X_pca,
    columns=pca_columns
)

pca_df.insert(
    0,
    "News_ID",
    df["News_ID"]
)


# ==========================================================
# 6. SAVE PCA DATA
# ==========================================================

os.makedirs(
    "output",
    exist_ok=True
)

output_file = (
    "output/news_pca.csv"
)

pca_df.to_csv(
    output_file,
    index=False
)


# ==========================================================
# 7. REPORT
# ==========================================================

print("\n" + "=" * 70)
print("PCA COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nOriginal Shape:",
    X_embeddings.shape
)

print(
    "PCA Shape:",
    X_pca.shape
)

print(
    "Explained Variance:",
    round(
        total_variance * 100,
        2
    ),
    "%"
)

print(
    "\nSaved File:",
    output_file
)

print("\n" + "=" * 70)
print("STEP 07 COMPLETED")
print("=" * 70)