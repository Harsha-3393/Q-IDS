import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

# ------------------
# Load
# ------------------

df = pd.read_csv(
    "datasets/processed.csv"
)

print("Loaded")

# ------------------
# Target
# ------------------

TARGET = "label"

REMOVE = [
    TARGET,
    "attack_cat"
]

REMOVE = [
    c
    for c in REMOVE
    if c in df.columns
]

X = df.drop(
    columns=REMOVE
)

y = df[TARGET]

# ------------------
# Scaling
# ------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ------------------
# Feature Selection
# ------------------

selector = SelectKBest(
    score_func=mutual_info_classif,
    k=20
)

X_selected = selector.fit_transform(
    X_scaled,
    y
)

selected_columns = X.columns[
    selector.get_support()
]

print("\nTop Features:\n")

for c in selected_columns:
    print(c)

# ------------------
# Save
# ------------------

final = pd.DataFrame(
    X_selected,
    columns=selected_columns
)

final["label"] = y

final.to_csv(
    "datasets/final_dataset.csv",
    index=False
)

print("\nSaved final_dataset.csv")
print(final.shape)