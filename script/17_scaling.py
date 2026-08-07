import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler


print("=" * 70)
print("STEP 15 - FEATURE SCALING")
print("=" * 70)


# ==========================================================
# 1. LOAD PCA DATA
# ==========================================================

train_file = "output/train_pca.csv"
test_file = "output/test_pca.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)


print("\nTraining Shape:", train_df.shape)
print("Testing Shape:", test_df.shape)


# ==========================================================
# 2. SEPARATE FEATURES
# ==========================================================

feature_columns = [
    column
    for column in train_df.columns
    if column not in ["Category", "Title"]
]


X_train = train_df[feature_columns]
X_test = test_df[feature_columns]

y_train = train_df["Category"]
y_test = test_df["Category"]

title_train = train_df["Title"]
title_test = test_df["Title"]


# ==========================================================
# 3. SCALER
# ==========================================================

scaler = StandardScaler()


print("\nFitting scaler on TRAINING data only...")

X_train_scaled = scaler.fit_transform(
    X_train
)


print("Transforming TEST data...")

X_test_scaled = scaler.transform(
    X_test
)


# ==========================================================
# 4. DATAFRAMES
# ==========================================================

train_scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=feature_columns
)

test_scaled_df = pd.DataFrame(
    X_test_scaled,
    columns=feature_columns
)


train_scaled_df["Category"] = y_train.values
train_scaled_df["Title"] = title_train.values

test_scaled_df["Category"] = y_test.values
test_scaled_df["Title"] = title_test.values


# ==========================================================
# 5. SAVE
# ==========================================================

train_output = "output/train_pca_scaled.csv"
test_output = "output/test_pca_scaled.csv"

train_scaled_df.to_csv(
    train_output,
    index=False
)

test_scaled_df.to_csv(
    test_output,
    index=False
)


# Save scaler
with open(
    "output/scaler_new.pkl",
    "wb"
) as f:

    pickle.dump(
        scaler,
        f
    )


print("\n" + "=" * 70)
print("SCALING COMPLETED")
print("=" * 70)

print("\nTrain:", train_scaled_df.shape)
print("Test :", test_scaled_df.shape)

print("\nSaved:")
print(train_output)
print(test_output)
print("output/scaler_new.pkl")