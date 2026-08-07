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
    f1_score
)


print("=" * 70)
print("STEP 18 - FINAL 10 MULTICLASS MODELS")
print("=" * 70)


# ==========================================================
# 1. LOAD DATA
# ==========================================================

train_df = pd.read_csv(
    "output/train_final_smote.csv"
)

test_df = pd.read_csv(
    "output/test_final_untouched.csv"
)


print("\nTraining Shape:")
print(train_df.shape)

print("\nTesting Shape:")
print(test_df.shape)


# ==========================================================
# 2. FEATURES
# ==========================================================

feature_columns = [
    col
    for col in train_df.columns
    if col != "Category"
]


X_train = train_df[feature_columns]
y_train = train_df["Category"]

X_test = test_df[feature_columns]
y_test = test_df["Category"]


print("\nFeature Shape:")
print(X_train.shape)

print("\nClasses:")
print(sorted(y_train.unique()))


# ==========================================================
# 3. CHECK FEATURES
# ==========================================================

print("\nMissing Training Values:")
print(X_train.isna().sum().sum())

print("Missing Testing Values:")
print(X_test.isna().sum().sum())

print("Infinite Training Values:")
print(np.isinf(X_train.select_dtypes(include=np.number)).sum().sum())

print("Infinite Testing Values:")
print(np.isinf(X_test.select_dtypes(include=np.number)).sum().sum())


# ==========================================================
# 4. CLASSES
# ==========================================================

classes = sorted(
    y_train.unique()
)

class_to_number = {
    name: i
    for i, name in enumerate(classes)
}

number_to_class = {
    i: name
    for name, i in class_to_number.items()
}


# ==========================================================
# 5. XGBOOST TARGET
# ==========================================================

y_train_xgb = y_train.map(
    class_to_number
)


# ==========================================================
# 6. 10 MULTICLASS MODELS
# ==========================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            C=2,
            max_iter=3000,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            n_jobs=-1
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=30,
            min_samples_split=2,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42
        ),

    "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42
        ),

    "SVM RBF":
        SVC(
            C=5,
            gamma="scale",
            random_state=42
        ),

    "Linear SVM":
        LinearSVC(
            C=1,
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
            learning_rate=0.05,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softmax",
            num_class=len(classes),
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        )
}


# ==========================================================
# 7. TRAINING
# ==========================================================

results = []

all_predictions = {}


for model_name, model in models.items():

    print("\n")
    print("=" * 70)
    print("TRAINING:", model_name)
    print("=" * 70)

    try:

        if model_name == "XGBoost":

            model.fit(
                X_train,
                y_train_xgb
            )

            encoded_predictions = (
                model.predict(X_test)
            )

            predictions = np.array([
                number_to_class[int(x)]
                for x in encoded_predictions
            ])

        else:

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )


        # ==================================================
        # METRICS
        # ==================================================

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


        print(
            "Accuracy:",
            round(accuracy * 100, 2),
            "%"
        )

        print(
            "Precision:",
            round(precision, 4)
        )

        print(
            "Recall:",
            round(recall, 4)
        )

        print(
            "F1:",
            round(f1, 4)
        )


    except Exception as e:

        print(
            "\nERROR in",
            model_name
        )

        print(
            str(e)
        )


# ==========================================================
# 8. RESULTS
# ==========================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    "Accuracy",
    ascending=False
)


print("\n")
print("=" * 70)
print("FINAL 10 MODEL RESULTS")
print("=" * 70)


display_df = results_df.copy()

display_df["Accuracy"] = (
    display_df["Accuracy"] * 100
).round(2)


print(
    display_df.to_string(
        index=False
    )
)


# ==========================================================
# 9. SAVE RESULTS
# ==========================================================

results_df.to_csv(
    "output/final_10_model_results.csv",
    index=False
)


# ==========================================================
# 10. BEST MODEL
# ==========================================================

best_model_name = (
    results_df.iloc[0]["Model"]
)

best_predictions = all_predictions[
    best_model_name
]


print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)


print(
    "Model:",
    best_model_name
)

print(
    "Accuracy:",
    round(
        results_df.iloc[0]["Accuracy"] * 100,
        2
    ),
    "%"
)


# ==========================================================
# 11. FINAL PREDICTION CSV
# ==========================================================

if "Title" in test_df.columns:

    titles = test_df["Title"].values

else:

    titles = [""] * len(test_df)


prediction_df = pd.DataFrame({

    "Title": titles,

    "Actual_Value":
        y_test.values,

    "Predicted_Value":
        best_predictions

})


prediction_df.to_csv(
    "output/final_test_predictions.csv",
    index=False,
    encoding="utf-8"
)


print("\nSaved:")
print(
    "output/final_10_model_results.csv"
)

print(
    "output/final_test_predictions.csv"
)


print("\n")
print("=" * 70)
print("STEP 18 COMPLETED SUCCESSFULLY")
print("=" * 70)