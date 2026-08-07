import pandas as pd


print("=" * 70)
print("STEP 19 - MODEL PERFORMANCE COMPARISON")
print("=" * 70)


# ==========================================================
# 1. LOAD BASELINE
# ==========================================================

baseline = pd.read_csv(
    "output/model_comparison_baseline.csv"
)


baseline["Experiment"] = "Baseline"


# ==========================================================
# 2. LOAD OLD SMOTE
# ==========================================================

old_smote = pd.read_csv(
    "output/model_comparison_smote.csv"
)


old_smote["Experiment"] = "Old SMOTE"


# ==========================================================
# 3. LOAD FINAL PIPELINE
# ==========================================================

final_results = pd.read_csv(
    "output/final_10_model_results.csv"
)


final_results["Experiment"] = (
    "TF-IDF + Embeddings + PCA + SMOTE"
)


# ==========================================================
# 4. STANDARDIZE COLUMN NAMES
# ==========================================================

for df in [
    baseline,
    old_smote,
    final_results
]:

    if "Accuracy" not in df.columns:

        raise ValueError(
            "Accuracy column missing."
        )


# ==========================================================
# 5. COMBINE
# ==========================================================

comparison = pd.concat(

    [
        baseline[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1_Score",
                "Experiment"
            ]
        ],

        old_smote[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1_Score",
                "Experiment"
            ]
        ],

        final_results[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1_Score",
                "Experiment"
            ]
        ]

    ],

    ignore_index=True

)


# ==========================================================
# 6. PERCENTAGE
# ==========================================================

comparison["Accuracy_Percent"] = (
    comparison["Accuracy"] * 100
).round(2)


# ==========================================================
# 7. SAVE
# ==========================================================

comparison.to_csv(
    "output/all_experiments_comparison.csv",
    index=False
)


# ==========================================================
# 8. BEST OVERALL
# ==========================================================

best = comparison.sort_values(
    "Accuracy",
    ascending=False
).iloc[0]


print("\n" + "=" * 70)
print("BEST OVERALL RESULT")
print("=" * 70)

print(
    "\nModel:",
    best["Model"]
)

print(
    "Experiment:",
    best["Experiment"]
)

print(
    "Accuracy:",
    best["Accuracy_Percent"],
    "%"
)


print("\nSaved:")
print(
    "output/all_experiments_comparison.csv"
)


print("\n" + "=" * 70)