import pandas as pd
import numpy as np
import os

from imblearn.over_sampling import SMOTE


print("=" * 70)
print("STEP 10 - MULTICLASS SMOTE AUGMENTATION")
print("=" * 70)


# ==========================================================
# 1. LOAD ORIGINAL SCALED TRAINING DATA
# ==========================================================

train_file = "output/train_scaled.csv"
test_file = "output/test_scaled.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)


print("\nOriginal Training Shape:")
print(train_df.shape)

print("\nOriginal Testing Shape:")
print(test_df.shape)


# ==========================================================
# 2. TARGET COLUMN
# ==========================================================

target_column = "Category"


if target_column not in train_df.columns:

    raise ValueError(
        "Category column not found."
    )


# ==========================================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================================

X_train = train_df.drop(
    columns=[target_column]
)

y_train = train_df[target_column]


# ==========================================================
# 4. SHOW ORIGINAL DISTRIBUTION
# ==========================================================

print("\n" + "=" * 70)
print("BEFORE SMOTE")
print("=" * 70)

original_counts = (
    y_train.value_counts()
    .sort_index()
)

print(
    original_counts
)


# ==========================================================
# 5. DEFINE ALL SIX CLASSES
# ==========================================================

target_classes = [

    "Business",
    "Energy",
    "Health",
    "Markets",
    "Politics",
    "Technology"

]


print("\nTarget classes:")

for class_name in target_classes:

    print(
        f"{class_name}: 1500"
    )


# ==========================================================
# 6. CHECK CLASSES
# ==========================================================

actual_classes = set(
    y_train.unique()
)

required_classes = set(
    target_classes
)


missing_classes = (
    required_classes
    - actual_classes
)


if missing_classes:

    raise ValueError(
        "These classes are missing from training data: "
        + str(missing_classes)
    )


# ==========================================================
# 7. CHECK CURRENT COUNTS
# ==========================================================

print("\n" + "=" * 70)
print("CURRENT COUNTS")
print("=" * 70)


for class_name in target_classes:

    count = (
        y_train == class_name
    ).sum()

    print(
        f"{class_name}: {count}"
    )


# ==========================================================
# 8. CREATE SMOTE TARGET
# ==========================================================

sampling_strategy = {}

for class_name in target_classes:

    current_count = (
        y_train == class_name
    ).sum()

    # Only augment classes below 1500
    if current_count < 1500:

        sampling_strategy[
            class_name
        ] = 1500


# ==========================================================
# 9. CHECK IF AUGMENTATION IS REQUIRED
# ==========================================================

if len(sampling_strategy) == 0:

    print(
        "\nAll classes already contain "
        "1500 or more samples."
    )

    X_train_smote = X_train.copy()

    y_train_smote = y_train.copy()

else:

    print("\n" + "=" * 70)
    print("SMOTE TARGET")
    print("=" * 70)

    print(
        sampling_strategy
    )


    # ======================================================
    # FIND SAFE K VALUE
    # ======================================================

    minimum_class_count = min(

        (
            y_train == class_name
        ).sum()

        for class_name in target_classes

    )


    if minimum_class_count < 2:

        raise ValueError(
            "A class contains fewer than "
            "2 samples. SMOTE cannot be applied."
        )


    k_neighbors = min(
        5,
        minimum_class_count - 1
    )


    print(
        "\nSMOTE k_neighbors:",
        k_neighbors
    )


    # ======================================================
    # APPLY SMOTE
    # ======================================================

    print("\nApplying SMOTE...")

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
# 10. CONVERT TO DATAFRAME
# ==========================================================

augmented_train_df = pd.DataFrame(

    X_train_smote,

    columns=X_train.columns

)


augmented_train_df[
    target_column
] = y_train_smote.values


# ==========================================================
# 11. AFTER SMOTE DISTRIBUTION
# ==========================================================

print("\n" + "=" * 70)
print("AFTER SMOTE")
print("=" * 70)


final_counts = (

    augmented_train_df[
        target_column
    ]
    .value_counts()
    .sort_index()

)


print(
    final_counts
)


# ==========================================================
# 12. VERIFY EACH CLASS
# ==========================================================

print("\n" + "=" * 70)
print("TARGET VERIFICATION")
print("=" * 70)


all_correct = True


for class_name in target_classes:

    final_count = (

        augmented_train_df[
            target_column
        ] == class_name

    ).sum()


    print(
        f"{class_name}: {final_count}"
    )


    if final_count != 1500:

        all_correct = False


# ==========================================================
# 13. FINAL CHECK
# ==========================================================

if not all_correct:

    raise ValueError(
        "Not all classes reached exactly 1500 samples."
    )


total_samples = len(
    augmented_train_df
)


print(
    "\nTotal Augmented Training Samples:",
    total_samples
)


if total_samples != 9000:

    raise ValueError(
        "Expected 9000 training samples "
        "but got "
        + str(total_samples)
    )


# ==========================================================
# 14. SAVE AUGMENTED TRAINING DATA
# ==========================================================

output_file = (
    "output/train_smote_1500.csv"
)


augmented_train_df.to_csv(

    output_file,

    index=False

)


# ==========================================================
# 15. SAVE ORIGINAL TEST DATA
# ==========================================================

test_output_file = (
    "output/test_smote_1500.csv"
)


test_df.to_csv(

    test_output_file,

    index=False

)


# ==========================================================
# 16. SAVE CLASS DISTRIBUTION
# ==========================================================

distribution_df = pd.DataFrame({

    "Category":
        target_classes,

    "Before_SMOTE": [

        (
            y_train == class_name
        ).sum()

        for class_name
        in target_classes

    ],

    "After_SMOTE": [

        (
            y_train_smote == class_name
        ).sum()

        for class_name
        in target_classes

    ]

})


distribution_file = (
    "output/smote_class_distribution.csv"
)


distribution_df.to_csv(

    distribution_file,

    index=False

)


# ==========================================================
# 17. FINAL REPORT
# ==========================================================

print("\n" + "=" * 70)
print("SMOTE AUGMENTATION COMPLETED SUCCESSFULLY")
print("=" * 70)


print(
    "\nOriginal Training:",
    len(train_df)
)


print(
    "Augmented Training:",
    len(augmented_train_df)
)


print(
    "Testing:",
    len(test_df)
)


print(
    "\nEach class:",
    1500,
    "samples"
)


print(
    "Total Training:",
    9000
)


print(
    "\nAugmented Training File:"
)

print(
    output_file
)


print(
    "\nTest File:"
)

print(
    test_output_file
)


print(
    "\nDistribution File:"
)

print(
    distribution_file
)


print("\n" + "=" * 70)
print("STEP 10 COMPLETED")
print("=" * 70)