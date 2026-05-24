import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# -------------------
# Load
# -------------------

df = pd.read_csv(
    "datasets/final_dataset.csv"
)

print("Dataset Loaded")

# -------------------
# Remove Leakage
# -------------------

remove_cols = []

if "attack_cat" in df.columns:
    remove_cols.append("attack_cat")

if len(remove_cols):
    df = df.drop(
        columns=remove_cols
    )

# -------------------
# Split
# -------------------

X = df.drop(
    columns=["label"]
)

y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------
# Scale
# -------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)

# -------------------
# Model
# -------------------

model = XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

print("\nTraining XGBoost...\n")

model.fit(
    X_train,
    y_train
)

# -------------------
# Predict
# -------------------

pred = model.predict(
    X_test
)

# -------------------
# Metrics
# -------------------

print("RESULTS\n")

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        pred
    )
)

print(
    "Precision:",
    precision_score(
        y_test,
        pred
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        pred
    )
)

print(
    "F1:",
    f1_score(
        y_test,
        pred
    )
)

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        pred
    )
)