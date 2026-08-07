import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier


print("=" * 70)
print("STEP 17 - HYPERPARAMETER TUNING")
print("=" * 70)


# ==========================================================
# 1. LOAD
# ==========================================================

train_df = pd.read_csv(
    "output/train_final_smote.csv"
)

test_df = pd.read_csv(
    "output/test_final_untouched.csv"
)


feature_columns = [
    column
    for column in train_df.columns
    if column != "Category"
]


X_train = train_df[feature_columns]
y_train = train_df["Category"]

X_test = test_df[feature_columns]
y_test = test_df["Category"]


# ==========================================================
# 2. ENCODE XGBOOST TARGET
# ==========================================================

classes = sorted(
    y_train.unique()
)

class_to_number = {
    name: i
    for i, name in enumerate(classes)
}

y_train_xgb = y_train.map(
    class_to_number
)

y_test_xgb = y_test.map(
    class_to_number
)


# ==========================================================
# 3. MODELS + PARAMETERS
# ==========================================================

searches = {}


searches["Logistic Regression"] = (
    LogisticRegression(
        max_iter=3000,
        random_state=42
    ),
    {
        "C": [
            0.1,
            0.5,
            1,
            2,
            5,
            10
        ]
    }
)


searches["Random Forest"] = (
    RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ),
    {
        "n_estimators": [
            300,
            500
        ],

        "max_depth": [
            None,
            20,
            30,
            40
        ],

        "min_samples_split": [
            2,
            5
        ],

        "max_features": [
            "sqrt",
            "log2"
        ]
    }
)


searches["Extra Trees"] = (
    ExtraTreesClassifier(
        random_state=42,
        n_jobs=-1
    ),
    {
        "n_estimators": [
            300,
            500
        ],

        "max_depth": [
            None,
            20,
            30,
            40
        ],

        "min_samples_split": [
            2,
            5
        ],

        "max_features": [
            "sqrt",
            "log2"
        ]
    }
)


searches["SVM RBF"] = (
    SVC(
        probability=True,
        random_state=42
    ),
    {
        "C": [
            0.5,
            1,
            2,
            5,
            10
        ],

        "gamma": [
            "scale",
            "auto"
        ]
    }
)


searches["XGBoost"] = (
    XGBClassifier(
        objective="multi:softmax",
        num_class=len(classes),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    ),
    {
        "n_estimators": [
            200,
            300,
            500
        ],

        "max_depth": [
            3,
            5,
            7
        ],

        "learning_rate": [
            0.03,
            0.05,
            0.1
        ],

        "subsample": [
            0.8,
            1.0
        ],

        "colsample_bytree": [
            0.8,
            1.0
        ]
    }
)


# ==========================================================
# 4. TUNING
# ==========================================================

os.makedirs(
    "output/tuned_models",
    exist_ok=True
)


tuned_results = []


for model_name, (
    model,
    parameters
) in searches.items():

    print("\n")
    print("=" * 70)
    print(
        "TUNING:",
        model_name
    )
    print("=" * 70)


    # XGBoost uses encoded labels
    if model_name == "XGBoost":

        search_X = X_train
        search_y = y_train_xgb

    else:

        search_X = X_train
        search_y = y_train


    search = RandomizedSearchCV(

        estimator=model,

        param_distributions=parameters,

        n_iter=8,

        cv=3,

        scoring="accuracy",

        random_state=42,

        n_jobs=-1,

        verbose=1

    )


    search.fit(
        search_X,
        search_y
    )


    best_model = (
        search.best_estimator_
    )


    if model_name == "XGBoost":

        predictions_encoded = (
            best_model.predict(X_test)
        )

        predictions = [
            classes[int(x)]
            for x in predictions_encoded
        ]

    else:

        predictions = (
            best_model.predict(X_test)
        )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    print(
        "\nBest CV Score:",
        round(
            search.best_score_ * 100,
            2
        ),
        "%"
    )

    print(
        "Test Accuracy:",
        round(
            accuracy * 100,
            2
        ),
        "%"
    )

    print(
        "Best Parameters:"
    )

    print(
        search.best_params_
    )


    tuned_results.append({

        "Model": model_name,

        "CV_Accuracy":
            search.best_score_,

        "Test_Accuracy":
            accuracy,

        "Best_Parameters":
            str(search.best_params_)

    })


    # Save model

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )


    with open(
        f"output/tuned_models/{safe_name}.pkl",
        "wb"
    ) as f:

        pickle.dump(
            best_model,
            f
        )


# ==========================================================
# 5. SAVE RESULTS
# ==========================================================

results_df = pd.DataFrame(
    tuned_results
)

results_df = results_df.sort_values(
    "Test_Accuracy",
    ascending=False
)


results_df.to_csv(
    "output/hyperparameter_tuning_results.csv",
    index=False
)


print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING COMPLETED")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)