import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

df = pd.read_csv(
    "datasets/final_dataset.csv"
)

X = df.drop(
    columns=["label"]
)

y = df["label"]

# Remove dominant feature
if "sttl" in X.columns:
    X = X.drop(
        columns=["sttl"]
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

pred = model.predict(
    X_test
)

print(
    "Accuracy Without sttl:",
    accuracy_score(
        y_test,
        pred
    )
)