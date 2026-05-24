import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load
df = pd.read_csv(
    "datasets/final_dataset.csv"
)

X = df.drop(
    columns=["label"]
)

y = df["label"]

# Train
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

print("Training...")

model.fit(
    X_train,
    y_train
)

# Save
joblib.dump(
    model,
    "models/xgb_model.pkl"
)

print("Saved")