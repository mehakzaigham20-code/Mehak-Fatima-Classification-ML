import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


print("=" * 70)
print("STEP 08 - FEATURE COMBINATION, SPLIT & SCALING")
print("=" * 70)


# ==========================================================
# 1. LOAD FEATURE ENGINEERED DATA
# ==========================================================

featured_file = "output/featured_news.csv"

df = pd.read_csv(
    featured_file
)

print("\nFeatured Dataset Shape:")
print(df.shape)


# ==========================================================
# 2. LOAD PCA DATA
# ==========================================================

pca_file = "output/news_pca.csv"

pca_df = pd.read_csv(
    pca_file
)

print("\nPCA Dataset Shape:")
print(pca_df.shape)


# ==========================================================
# 3. CHECK ROW COUNTS
# ==========================================================

if len(df) != len(pca_df):

    raise ValueError(
        "Featured dataset and PCA dataset "
        "have different number of rows."
    )


print(
    "\nRow count check: PASSED"
)


# ==========================================================
# 4. ENGINEERED FEATURES
# EXACTLY 15
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
# 5. PCA FEATURES
# ==========================================================

pca_features = [

    column
    for column in pca_df.columns
    if column.startswith("PCA_")
]


print(
    "\nEngineered Features:",
    len(engineered_features)
)

print(
    "PCA Features:",
    len(pca_features)
)


# ==========================================================
# 6. CHECK FEATURES
# ==========================================================

missing_engineered = [

    feature
    for feature in engineered_features
    if feature not in df.columns
]

if missing_engineered:

    raise ValueError(
        "Missing engineered features: "
        + str(missing_engineered)
    )


if len(pca_features) == 0:

    raise ValueError(
        "No PCA features found."
    )


# ==========================================================
# 7. COMBINE FEATURES
# ==========================================================

X_engineered = df[
    engineered_features
].copy()

X_pca = pca_df[
    pca_features
].copy()


# Reset indexes
X_engineered = X_engineered.reset_index(
    drop=True
)

X_pca = X_pca.reset_index(
    drop=True
)


# Combine
X = pd.concat(
    [
        X_engineered,
        X_pca
    ],
    axis=1
)


# ==========================================================
# 8. TARGET
# ==========================================================

if "Category" not in df.columns:

    raise ValueError(
        "Category column not found."
    )


y = df[
    "Category"
].copy()


# ==========================================================
# 9. CLEAN FEATURES
# ==========================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)


# Convert all features to numeric
for column in X.columns:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

X = X.fillna(0)


print("\nCombined Feature Matrix:")
print(X.shape)

print(
    "Total Features:",
    X.shape[1]
)


# ==========================================================
# 10. TRAIN / TEST SPLIT
# 80% TRAIN - 20% TEST
# ==========================================================

print("\nCreating 80/20 train-test split...")


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nSplit completed.")

print(
    "Training Samples:",
    X_train.shape[0]
)

print(
    "Testing Samples:",
    X_test.shape[0]
)


# ==========================================================
# 11. SCALING
# ==========================================================

print("\nScaling features...")


scaler = StandardScaler()


# IMPORTANT:
# Fit ONLY on training data

X_train_scaled = scaler.fit_transform(
    X_train
)


# Test data only transformed
X_test_scaled = scaler.transform(
    X_test
)


print(
    "Scaling completed."
)


# ==========================================================
# 12. CONVERT BACK TO DATAFRAME
# ==========================================================

X_train_scaled = pd.DataFrame(

    X_train_scaled,

    columns=X.columns,

    index=X_train.index
)


X_test_scaled = pd.DataFrame(

    X_test_scaled,

    columns=X.columns,

    index=X_test.index
)


# ==========================================================
# 13. SAVE TRAINING DATA
# ==========================================================

os.makedirs(
    "output",
    exist_ok=True
)


train_data = X_train_scaled.copy()

train_data["Category"] = y_train


train_file = (
    "output/train_scaled.csv"
)


train_data.to_csv(
    train_file,
    index=False
)


# ==========================================================
# 14. SAVE TEST DATA
# ==========================================================

test_data = X_test_scaled.copy()

test_data["Category"] = y_test


test_file = (
    "output/test_scaled.csv"
)


test_data.to_csv(
    test_file,
    index=False
)


# ==========================================================
# 15. SAVE ORIGINAL TRAIN/TEST INFORMATION
# ==========================================================

train_original = df.loc[
    X_train.index
].copy()


test_original = df.loc[
    X_test.index
].copy()


train_original.to_csv(
    "output/train_original.csv",
    index=False
)


test_original.to_csv(
    "output/test_original.csv",
    index=False
)


# ==========================================================
# 16. SAVE SCALER
# ==========================================================

import joblib


scaler_file = (
    "output/scaler.pkl"
)


joblib.dump(
    scaler,
    scaler_file
)


# ==========================================================
# 17. FINAL REPORT
# ==========================================================

print("\n" + "=" * 70)
print("SCALING COMPLETED SUCCESSFULLY")
print("=" * 70)


print(
    "\nTotal Dataset:",
    len(df)
)


print(
    "Training Samples:",
    len(X_train)
)


print(
    "Testing Samples:",
    len(X_test)
)


print(
    "\nTraining Percentage:",
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


print(
    "\nTotal Combined Features:",
    X.shape[1]
)


print(
    "\nTraining File:"
)

print(
    train_file
)


print(
    "\nTesting File:"
)

print(
    test_file
)


print(
    "\nScaler File:"
)

print(
    scaler_file
)


print("\n" + "=" * 70)
print("STEP 08 COMPLETED")
print("=" * 70)