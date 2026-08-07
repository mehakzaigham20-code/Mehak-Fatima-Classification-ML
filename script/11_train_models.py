import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.svm import SVC, LinearSVC

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================================
# STEP 09 - BASELINE MULTICLASS MODEL TRAINING
# ==========================================================

print("=" * 70)
print("STEP 09 - BASELINE MULTICLASS MODEL TRAINING")
print("=" * 70)


# ==========================================================
# 1. LOAD DATA
# ==========================================================

train_file = "output/train_scaled.csv"
test_file = "output/test_scaled.csv"
original_test_file = "output/test_original.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
original_test_df = pd.read_csv(original_test_file)

print("\nTraining Dataset Shape:")
print(train_df.shape)

print("\nTesting Dataset Shape:")
print(test_df.shape)

print("\nOriginal Test Dataset Shape:")
print(original_test_df.shape)


# ==========================================================
# 2. CHECK ROW COUNTS
# ==========================================================

if len(test_df) != len(original_test_df):

    raise ValueError(
        "Scaled test data and original test data "
        "have different row counts."
    )


# ==========================================================
# 3. TARGET
# ==========================================================

target_column = "Category"

if target_column not in train_df.columns:
    raise ValueError(
        "Category column not found in training data."
    )

if target_column not in test_df.columns:
    raise ValueError(
        "Category column not found in testing data."
    )


# ==========================================================
# 4. FEATURES AND TARGET
# ==========================================================

X_train = train_df.drop(
    columns=[target_column]
)

y_train = train_df[target_column]

X_test = test_df.drop(
    columns=[target_column]
)

y_test = test_df[target_column]


print("\nFeature Shapes:")

print(
    "X_train:",
    X_train.shape
)

print(
    "X_test:",
    X_test.shape
)


# ==========================================================
# 5. MULTICLASS INFORMATION
# ==========================================================

classes = sorted(
    y_train.unique()
)

number_of_classes = len(classes)

print("\n" + "=" * 70)
print("TARGET INFORMATION")
print("=" * 70)

print(
    "\nNumber of Classes:",
    number_of_classes
)

print("\nClasses:")

for number, class_name in enumerate(
    classes,
    start=1
):

    print(
        f"{number}. {class_name}"
    )


if number_of_classes < 3:

    raise ValueError(
        "This project requires multiclass classification."
    )


# ==========================================================
# 6. ENCODE TARGET
# ==========================================================

class_to_number = {

    class_name: number

    for number, class_name
    in enumerate(classes)
}


number_to_class = {

    number: class_name

    for class_name, number
    in class_to_number.items()
}


y_train_encoded = y_train.map(
    class_to_number
)

y_test_encoded = y_test.map(
    class_to_number
)


# ==========================================================
# 7. OUTPUT FOLDER
# ==========================================================

os.makedirs(
    "output/confusion_matrices",
    exist_ok=True
)


# ==========================================================
# 8. 10 MULTICLASS MODELS
# ==========================================================

models = {

    # 1
    "Logistic Regression":

        LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            random_state=42
        ),


    # 2
    "KNN":

        KNeighborsClassifier(
            n_neighbors=7,
            weights="distance"
        ),


    # 3
    "Decision Tree":

        DecisionTreeClassifier(
            max_depth=20,
            random_state=42,
            class_weight="balanced"
        ),


    # 4
    "Random Forest":

        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),


    # 5
    "Extra Trees":

        ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),


    # 6
    "SVM RBF":

        SVC(
            kernel="rbf",
            C=2,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42
        ),


    # 7
    "Linear SVM":

        LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=10000,
            random_state=42
        ),


    # 8
    "Gradient Boosting":

        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),


    # 9
    "AdaBoost":

        AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.5,
            random_state=42
        ),


    # 10
    "XGBoost":

        XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softmax",
            num_class=number_of_classes,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        )
}


# ==========================================================
# 9. TRAINING
# ==========================================================

results = []

all_predictions = {}


for model_name, model in models.items():

    print("\n")
    print("=" * 70)
    print(
        "TRAINING:",
        model_name
    )
    print("=" * 70)


    # ------------------------------------------------------
    # XGBOOST
    # ------------------------------------------------------

    if model_name == "XGBoost":

        model.fit(
            X_train,
            y_train_encoded
        )

        prediction_encoded = model.predict(
            X_test
        )

        predictions_test = np.array([

            number_to_class[
                int(prediction)
            ]

            for prediction
            in prediction_encoded

        ])


    # ------------------------------------------------------
    # OTHER MODELS
    # ------------------------------------------------------

    else:

        model.fit(
            X_train,
            y_train
        )

        predictions_test = model.predict(
            X_test
        )


    # ======================================================
    # METRICS
    # ======================================================

    accuracy = accuracy_score(
        y_test,
        predictions_test
    )

    precision = precision_score(
        y_test,
        predictions_test,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions_test,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions_test,
        average="weighted",
        zero_division=0
    )


    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1

    })


    all_predictions[
        model_name
    ] = predictions_test


    # ======================================================
    # DISPLAY METRICS
    # ======================================================

    print(
        "\nAccuracy:",
        round(
            accuracy * 100,
            2
        ),
        "%"
    )

    print(
        "Precision:",
        round(
            precision,
            4
        )
    )

    print(
        "Recall:",
        round(
            recall,
            4
        )
    )

    print(
        "F1 Score:",
        round(
            f1,
            4
        )
    )


    # ======================================================
    # CLASSIFICATION REPORT
    # ======================================================

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions_test,
            labels=classes,
            zero_division=0
        )
    )


    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    cm = confusion_matrix(
        y_test,
        predictions_test,
        labels=classes
    )


    plt.figure(
        figsize=(10, 8)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.xlabel(
        "Predicted Value"
    )

    plt.ylabel(
        "Actual Value"
    )

    plt.tight_layout()


    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


    confusion_file = (
        f"output/confusion_matrices/"
        f"{safe_name}.png"
    )


    plt.savefig(
        confusion_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================================
# 10. MODEL COMPARISON
# ==========================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)


results_file = (
    "output/model_comparison_baseline.csv"
)


results_df.to_csv(
    results_file,
    index=False
)


print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


display_results = results_df.copy()


display_results["Accuracy"] = (
    display_results["Accuracy"] * 100
).round(2)


display_results["Precision"] = (
    display_results["Precision"]
).round(4)


display_results["Recall"] = (
    display_results["Recall"]
).round(4)


display_results["F1_Score"] = (
    display_results["F1_Score"]
).round(4)


print(
    display_results.to_string(
        index=False
    )
)


# ==========================================================
# 11. BEST MODEL
# ==========================================================

best_model_row = (
    results_df.iloc[0]
)

best_model_name = (
    best_model_row["Model"]
)

best_accuracy = (
    best_model_row["Accuracy"]
)


print("\n")
print("=" * 70)
print("BEST BASELINE MODEL")
print("=" * 70)


print(
    "\nBest Model:",
    best_model_name
)

print(
    "Accuracy:",
    round(
        best_accuracy * 100,
        2
    ),
    "%"
)

print(
    "Precision:",
    round(
        best_model_row["Precision"],
        4
    )
)

print(
    "Recall:",
    round(
        best_model_row["Recall"],
        4
    )
)

print(
    "F1 Score:",
    round(
        best_model_row["F1_Score"],
        4
    )
)


# ==========================================================
# 12. FINAL TEST PREDICTIONS
# ==========================================================

best_predictions = (
    all_predictions[
        best_model_name
    ]
)


# Get actual article titles
if "Title" not in original_test_df.columns:

    raise ValueError(
        "Title column not found in test_original.csv"
    )


prediction_df = pd.DataFrame({

    "Title":
        original_test_df[
            "Title"
        ].values,

    "Actual_Value":
        y_test.values,

    "Predicted_Value":
        best_predictions

})


# ==========================================================
# 13. SAVE PREDICTIONS
# ==========================================================

prediction_file = (
    "output/baseline_test_predictions.csv"
)


prediction_df.to_csv(
    prediction_file,
    index=False,
    encoding="utf-8"
)


# ==========================================================
# 14. FINAL REPORT
# ==========================================================

print("\n")
print("=" * 70)
print("BASELINE CLASSIFICATION COMPLETED")
print("=" * 70)

print(
    "\nTotal Models:",
    len(models)
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
    "Number of Classes:",
    number_of_classes
)

print(
    "\nBest Model:",
    best_model_name
)

print(
    "Best Accuracy:",
    round(
        best_accuracy * 100,
        2
    ),
    "%"
)

print(
    "\nModel Comparison:"
)

print(
    results_file
)

print(
    "\nTest Predictions:"
)

print(
    prediction_file
)

print(
    "\nConfusion Matrices:"
)

print(
    "output/confusion_matrices/"
)

print("\n" + "=" * 70)
print("STEP 09 COMPLETED SUCCESSFULLY")
print("=" * 70)
