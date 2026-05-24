import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Load
df = pd.read_csv(
    "datasets/final_dataset.csv"
)

X = df.drop(
    columns=["label"]
)

y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
model = XGBClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# Importance
importance = model.feature_importances_

result = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

result = result.sort_values(
    by="Importance",
    ascending=False
)

print(result)

# Plot
plt.figure(
    figsize=(12,8)
)

plt.barh(
    result["Feature"][:10],
    result["Importance"][:10]
)

plt.xlabel(
    "Importance"
)

plt.title(
    "Top Security Features"
)

plt.gca().invert_yaxis()

plt.show()