import pandas as pd
import numpy as np
import os

from sentence_transformers import SentenceTransformer


print("=" * 70)
print("STEP 06 - NEWS TEXT EMBEDDINGS")
print("=" * 70)


# ==========================================================
# 1. LOAD FEATURED DATA
# ==========================================================

input_file = "output/featured_news.csv"

df = pd.read_csv(input_file)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ==========================================================
# 2. CHECK TEXT COLUMN
# ==========================================================

if "Processed_Title" not in df.columns:

    raise ValueError(
        "Processed_Title column not found."
    )


# ==========================================================
# 3. PREPARE TEXT
# ==========================================================

texts = (
    df["Processed_Title"]
    .fillna("")
    .astype(str)
    .tolist()
)

print("\nTotal texts:")
print(len(texts))


# ==========================================================
# 4. LOAD SENTENCE TRANSFORMER
# ==========================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully.")


# ==========================================================
# 5. GENERATE EMBEDDINGS
# ==========================================================

print("\nGenerating embeddings...")
print("Please wait...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)


# ==========================================================
# 6. CHECK EMBEDDINGS
# ==========================================================

print("\nEmbedding generation completed.")

print(
    "Embedding Shape:",
    embeddings.shape
)

print(
    "Embedding Dimension:",
    embeddings.shape[1]
)


# ==========================================================
# 7. CREATE EMBEDDING DATAFRAME
# ==========================================================

embedding_columns = [
    f"Embedding_{i+1}"
    for i in range(
        embeddings.shape[1]
    )
]

embedding_df = pd.DataFrame(
    embeddings,
    columns=embedding_columns
)


# ==========================================================
# 8. ADD ID
# ==========================================================

embedding_df.insert(
    0,
    "News_ID",
    range(1, len(df) + 1)
)


# ==========================================================
# 9. SAVE EMBEDDINGS
# ==========================================================

os.makedirs(
    "output",
    exist_ok=True
)

embedding_file = (
    "output/news_embeddings.csv"
)

embedding_df.to_csv(
    embedding_file,
    index=False
)


# ==========================================================
# 10. SAVE COMPLETE DATA + EMBEDDINGS
# ==========================================================

combined_df = pd.concat(
    [
        df.reset_index(drop=True),
        embedding_df.drop(
            columns=["News_ID"]
        ).reset_index(drop=True)
    ],
    axis=1
)

combined_file = (
    "output/news_with_embeddings.csv"
)

combined_df.to_csv(
    combined_file,
    index=False
)


# ==========================================================
# 11. FINAL REPORT
# ==========================================================

print("\n" + "=" * 70)
print("EMBEDDING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nOriginal Dataset Shape:",
    df.shape
)

print(
    "Embedding Matrix Shape:",
    embeddings.shape
)

print(
    "Embedding Dimensions:",
    embeddings.shape[1]
)

print(
    "\nEmbedding CSV:"
)

print(
    embedding_file
)

print(
    "\nCombined Dataset:"
)

print(
    combined_file
)

print("\nFirst 5 Embeddings:")

print(
    embedding_df.head()
)

print("\n" + "=" * 70)
print("STEP 06 COMPLETED")
print("=" * 70)