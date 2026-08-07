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


print("=" * 70)
print("STEP 11 - AUGMENTED MULTICLASS MODEL TRAINING & TESTING")
print("=" * 70)


# ==========================================================
# 1. LOAD AUGMENTED TRAINING DATA
# ==========================================================

train_file = "output/train_smote_1500.csv"
test_file = "output/test_smote_1500.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)


print("\nAugmented Training Shape:")
print(train_df.shape)

print("\nOriginal Testing Shape:")
print(test_df.shape)


# ==========================================================
# 2. LOAD ORIGINAL TEST DATA FOR TITLES
# ==========================================================

original_test_file = "output/test_original.csv"

original_test_df = pd.read_csv(
    original_test_file
)


if len(test_df) != len(original_test_df):

    raise ValueError(
        "Test dataset row count does not match "
        "original test dataset."
    )


# ==========================================================
# 3. TARGET
# ==========================================================

target = "Category"


# ==========================================================
# 4. FEATURES + TARGET
# ==========================================================

X_train = train_df.drop(
    columns=[target]
)

y_train = train_df[target]


X_test = test_df.drop(
    columns=[target]
)

y_test = test_df[target]


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
# 5. CHECK CLASSES
# ==========================================================

classes = sorted(
    y_train.unique()
)

number_of_classes = len(
    classes
)


print("\n" + "=" * 70)
print("MULTICLASS INFORMATION")
print("=" * 70)


print(
    "\nNumber of Classes:",
    number_of_classes
)


for number, class_name in enumerate(
    classes,
    start=1
):

    print(
        f"{number}. {class_name}"
    )


# ==========================================================
# 6. ENCODE TARGET FOR XGBOOST
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
# 7. CONFUSION MATRIX FOLDER
# ==========================================================

os.makedirs(
    "output/confusion_matrices_smote",
    exist_ok=True
)


# ==========================================================
# 8. 10 MULTICLASS MODELS
# ==========================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=7,
            weights="distance"
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=20,
            random_state=42,
            class_weight="balanced"
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),

    "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),

    "SVM RBF":
        SVC(
            kernel="rbf",
            C=2,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42
        ),

    "Linear SVM":
        LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=10000,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),

    "AdaBoost":
        AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.5,
            random_state=42
        ),

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
# 9. TRAIN + TEST
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

        predictions = np.array([

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

        predictions = model.predict(
            X_test
        )


    # ======================================================
    # METRICS
    # ======================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
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
    ] = predictions


    # ======================================================
    # PRINT RESULTS
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

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            labels=classes,
            zero_division=0
        )
    )


    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    cm = confusion_matrix(
        y_test,
        predictions,
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
        f"SMOTE Confusion Matrix - {model_name}"
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


    matrix_file = (
        "output/confusion_matrices_smote/"
        + safe_name
        + ".png"
    )


    plt.savefig(
        matrix_file,
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


smote_results_file = (
    "output/model_comparison_smote.csv"
)


results_df.to_csv(
    smote_results_file,
    index=False
)


print("\n")
print("=" * 70)
print("SMOTE MODEL COMPARISON")
print("=" * 70)


display_df = results_df.copy()


display_df["Accuracy"] = (
    display_df["Accuracy"] * 100
).round(2)


display_df["Precision"] = (
    display_df["Precision"]
).round(4)


display_df["Recall"] = (
    display_df["Recall"]
).round(4)


display_df["F1_Score"] = (
    display_df["F1_Score"]
).round(4)


print(
    display_df.to_string(
        index=False
    )
)


# ==========================================================
# 11. BEST SMOTE MODEL
# ==========================================================

best_row = results_df.iloc[0]

best_model = best_row["Model"]

best_accuracy = best_row["Accuracy"]


print("\n")
print("=" * 70)
print("BEST SMOTE MODEL")
print("=" * 70)


print(
    "\nBest Model:",
    best_model
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
        best_row["Precision"],
        4
    )
)

print(
    "Recall:",
    round(
        best_row["Recall"],
        4
    )
)

print(
    "F1 Score:",
    round(
        best_row["F1_Score"],
        4
    )
)


# ==========================================================
# 12. FINAL PREDICTION CSV
# ==========================================================

best_predictions = all_predictions[
    best_model
]


if "Title" not in original_test_df.columns:

    raise ValueError(
        "Title column not found."
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


prediction_file = (
    "output/smote_test_predictions.csv"
)


prediction_df.to_csv(
    prediction_file,
    index=False,
    encoding="utf-8"
)


# ==========================================================
# 13. FINAL REPORT
# ==========================================================

print("\n")
print("=" * 70)
print("STEP 11 COMPLETED SUCCESSFULLY")
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
    "Number of Models:",
    len(models)
)

print(
    "\nBest SMOTE Model:",
    best_model
)

print(
    "Best SMOTE Accuracy:",
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
    smote_results_file
)

print(
    "\nPredictions:"
)

print(
    prediction_file
)

print(
    "\nConfusion Matrices:"
)

print(
    "output/confusion_matrices_smote/"
)

print("\n" + "=" * 70)