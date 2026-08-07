import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("STEP 05 - CORRELATION ANALYSIS")
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
# 2. SELECT OUR 15 ENGINEERED FEATURES
# ==========================================================

feature_columns = [

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
# 3. CHECK FEATURES
# ==========================================================

missing_features = [
    feature
    for feature in feature_columns
    if feature not in df.columns
]

if missing_features:

    print("\nMissing Features:")
    print(missing_features)

    raise ValueError(
        "Some engineered features are missing."
    )


# ==========================================================
# 4. CREATE NUMERIC FEATURE DATA
# ==========================================================

X = df[feature_columns].copy()

for column in feature_columns:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)


print("\nFeature Matrix Shape:")
print(X.shape)


# ==========================================================
# 5. CORRELATION MATRIX
# ==========================================================

correlation_matrix = X.corr()

print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)

print(
    correlation_matrix.round(3)
)


# ==========================================================
# 6. SAVE CORRELATION MATRIX
# ==========================================================

os.makedirs(
    "output",
    exist_ok=True
)

correlation_matrix.to_csv(
    "output/correlation_matrix.csv"
)

print(
    "\nCorrelation matrix saved:"
)

print(
    "output/correlation_matrix.csv"
)


# ==========================================================
# 7. CORRELATION HEATMAP
# ==========================================================

plt.figure(
    figsize=(16, 12)
)

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title(
    "Feature Correlation Heatmap"
)

plt.tight_layout()

plt.savefig(
    "output/correlation_heatmap.png",
    dpi=300
)

plt.close()

print(
    "Correlation heatmap saved:"
)

print(
    "output/correlation_heatmap.png"
)


# ==========================================================
# 8. FIND HIGHLY CORRELATED FEATURES
# ==========================================================

print("\n" + "=" * 70)
print("HIGH CORRELATIONS")
print("=" * 70)


high_correlations = []

for i in range(
    len(feature_columns)
):

    for j in range(
        i + 1,
        len(feature_columns)
    ):

        feature_1 = feature_columns[i]
        feature_2 = feature_columns[j]

        correlation = correlation_matrix.loc[
            feature_1,
            feature_2
        ]

        if abs(correlation) >= 0.80:

            high_correlations.append(
                (
                    feature_1,
                    feature_2,
                    correlation
                )
            )


if high_correlations:

    for feature_1, feature_2, correlation in high_correlations:

        print(
            f"{feature_1} <-> {feature_2}: "
            f"{correlation:.3f}"
        )

else:

    print(
        "No feature pair has correlation >= 0.80."
    )


# ==========================================================
# 9. CORRELATION SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("CORRELATION SUMMARY")
print("=" * 70)

print(
    "\nTotal Features:",
    len(feature_columns)
)

print(
    "High Correlation Pairs:",
    len(high_correlations)
)

print(
    "\nCorrelation analysis completed successfully."
)

print("=" * 70)