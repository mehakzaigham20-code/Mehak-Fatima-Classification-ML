import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE


print("=" * 70)
print("STEP 16 - SMOTE AUGMENTATION")
print("=" * 70)


# ==========================================================
# 1. LOAD
# ==========================================================

train_file = "output/train_pca_scaled.csv"
test_file = "output/test_pca_scaled.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)


# ==========================================================
# 2. FEATURES
# ==========================================================

feature_columns = [
    column
    for column in train_df.columns
    if column not in ["Category", "Title"]
]


X_train = train_df[feature_columns]
y_train = train_df["Category"]


# Test remains untouched
X_test = test_df[feature_columns]
y_test = test_df["Category"]


print("\nBefore SMOTE:")
print(y_train.value_counts().sort_index())


# ==========================================================
# 3. TARGET = 1500 EACH
# ==========================================================

classes = [
    "Business",
    "Energy",
    "Health",
    "Markets",
    "Politics",
    "Technology"
]


sampling_strategy = {}

for class_name in classes:

    count = (
        y_train == class_name
    ).sum()

    if count < 1500:

        sampling_strategy[class_name] = 1500


print("\nSMOTE target:")
print(sampling_strategy)


# ==========================================================
# 4. SAFE K
# ==========================================================

minimum_count = min(
    (
        y_train == class_name
    ).sum()
    for class_name in classes
)


k_neighbors = min(
    5,
    minimum_count - 1
)


print(
    "\nK neighbors:",
    k_neighbors
)


# ==========================================================
# 5. SMOTE
# ==========================================================

smote = SMOTE(
    sampling_strategy=sampling_strategy,
    random_state=42,
    k_neighbors=k_neighbors
)


X_train_smote, y_train_smote = (
    smote.fit_resample(
        X_train,
        y_train
    )
)


# ==========================================================
# 6. SAVE
# ==========================================================

train_smote_df = pd.DataFrame(
    X_train_smote,
    columns=feature_columns
)

train_smote_df["Category"] = (
    y_train_smote.values
)


# Preserve untouched test set
test_save_df = test_df.copy()


train_output = (
    "output/train_final_smote.csv"
)

test_output = (
    "output/test_final_untouched.csv"
)


train_smote_df.to_csv(
    train_output,
    index=False
)

test_save_df.to_csv(
    test_output,
    index=False
)


# ==========================================================
# 7. REPORT
# ==========================================================

print("\n" + "=" * 70)
print("AFTER SMOTE")
print("=" * 70)

print(
    y_train_smote.value_counts()
    .sort_index()
)


print(
    "\nTotal Training:",
    len(train_smote_df)
)

print(
    "Total Testing:",
    len(test_save_df)
)


print("\nSaved:")
print(train_output)
print(test_output)

print("\n" + "=" * 70)