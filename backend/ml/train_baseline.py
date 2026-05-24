import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ------------------
# Load Dataset
# ------------------

df = pd.read_csv(
    "datasets/final_dataset.csv"
)

print("Dataset Loaded")

# ------------------
# Remove Data Leakage
# ------------------

if "attack_cat" in df.columns:
    df = df.drop(
        columns=["attack_cat"]
    )

# ------------------
# Split Features
# ------------------

X = df.drop(
    columns=["label"]
)

y = df["label"]

# ------------------
# Train Test Split
# ------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------
# Scaling
# ------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)

# ------------------
# Model
# ------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Model...\n")

model.fit(
    X_train,
    y_train
)

# ------------------
# Prediction
# ------------------

pred = model.predict(
    X_test
)

# ------------------
# Evaluation
# ------------------

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
    "F1 Score:",
    f1_score(
        y_test,
        pred
    )
)

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y_test,
        pred
    )
)