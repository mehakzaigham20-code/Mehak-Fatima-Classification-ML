import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


print("=" * 70)
print("STEP 14 - 80/20 SPLIT + PCA")
print("=" * 70)


# ==========================================================
# 1. LOAD COMBINED FEATURES
# ==========================================================

input_file = "output/combined_features.csv"

df = pd.read_csv(input_file)

print("\nInput Shape:")
print(df.shape)


# ==========================================================
# 2. REQUIRED COLUMNS
# ==========================================================

if "Category" not in df.columns:

    raise ValueError(
        "Category column not found."
    )

if "Title" not in df.columns:

    raise ValueError(
        "Title column not found."
    )


# ==========================================================
# 3. SEPARATE X AND Y
# ==========================================================

X = df.drop(
    columns=[
        "Category",
        "Title"
    ]
)

y = df["Category"]

titles = df["Title"]


print("\nFeature Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)


# ==========================================================
# 4. CHECK CLASSES
# ==========================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(
    y.value_counts()
    .sort_index()
)


# ==========================================================
# 5. 80/20 STRATIFIED SPLIT
# ==========================================================

print("\nCreating 80/20 split...")

(
    X_train,
    X_test,
    y_train,
    y_test,
    title_train,
    title_test
) = train_test_split(

    X,
    y,
    titles,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(
    "\nTraining Samples:",
    len(X_train)
)

print(
    "Testing Samples:",
    len(X_test)
)

print(
    "Training Percentage:",
    round(
        len(X_train) / len(df) * 100,
        2
    ),
    "%"
)

print(
    "Testing Percentage:",
    round(
        len(X_test) / len(df) * 100,
        2
    ),
    "%"
)


# ==========================================================
# 6. PCA
# ==========================================================

print("\n" + "=" * 70)
print("PCA")
print("=" * 70)


# Keep 95% explained variance
pca = PCA(
    n_components=0.95,
    svd_solver="full"
)


print(
    "\nFitting PCA on TRAINING DATA ONLY..."
)


X_train_pca = pca.fit_transform(
    X_train
)


print(
    "Transforming TEST DATA..."
)


X_test_pca = pca.transform(
    X_test
)


# ==========================================================
# 7. PCA INFORMATION
# ==========================================================

explained_variance = (
    pca.explained_variance_ratio_
)


total_variance = (
    explained_variance.sum()
)


print(
    "\nOriginal Features:",
    X_train.shape[1]
)

print(
    "PCA Components:",
    X_train_pca.shape[1]
)

print(
    "Explained Variance:",
    round(
        total_variance * 100,
        2
    ),
    "%"
)


# ==========================================================
# 8. CREATE PCA DATAFRAMES
# ==========================================================

pca_columns = [

    f"PC_{i+1}"

    for i in range(
        X_train_pca.shape[1]
    )

]


train_pca_df = pd.DataFrame(

    X_train_pca,

    columns=pca_columns

)


test_pca_df = pd.DataFrame(

    X_test_pca,

    columns=pca_columns

)


# ==========================================================
# 9. ADD TARGET
# ==========================================================

train_pca_df["Category"] = (
    y_train.values
)

test_pca_df["Category"] = (
    y_test.values
)


# ==========================================================
# 10. ADD TITLE
# ==========================================================

train_pca_df["Title"] = (
    title_train.values
)

test_pca_df["Title"] = (
    title_test.values
)


# ==========================================================
# 11. SAVE PCA DATA
# ==========================================================

train_file = (
    "output/train_pca.csv"
)

test_file = (
    "output/test_pca.csv"
)


train_pca_df.to_csv(
    train_file,
    index=False
)

test_pca_df.to_csv(
    test_file,
    index=False
)


# ==========================================================
# 12. SAVE PCA MODEL
# ==========================================================

pca_file = (
    "output/pca.pkl"
)


with open(
    pca_file,
    "wb"
) as file:

    pickle.dump(
        pca,
        file
    )


# ==========================================================
# 13. SAVE ORIGINAL SPLIT
# ==========================================================

train_original = pd.DataFrame({

    "Title":
        title_train.values,

    "Category":
        y_train.values

})


test_original = pd.DataFrame({

    "Title":
        title_test.values,

    "Category":
        y_test.values

})


train_original.to_csv(
    "output/train_original_new.csv",
    index=False
)

test_original.to_csv(
    "output/test_original_new.csv",
    index=False
)


# ==========================================================
# 14. FINAL REPORT
# ==========================================================

print("\n" + "=" * 70)
print("STEP 14 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nTraining Shape:",
    train_pca_df.shape
)

print(
    "Testing Shape:",
    test_pca_df.shape
)

print(
    "\nPCA Components:",
    X_train_pca.shape[1]
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
    "\nSaved Files:"
)

print(
    train_file
)

print(
    test_file
)

print(
    pca_file
)

print(
    "output/train_original_new.csv"
)

print(
    "output/test_original_new.csv"
)

print("\n" + "=" * 70)