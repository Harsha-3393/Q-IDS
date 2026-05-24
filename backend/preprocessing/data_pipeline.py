import pandas as pd
from sklearn.preprocessing import LabelEncoder

# -----------------
# Load Dataset
# -----------------

train = pd.read_csv("datasets/UNSW_NB15_training-set.csv")
test = pd.read_csv("datasets/UNSW_NB15_testing-set.csv")

print("Loaded Successfully")

# -----------------
# Merge
# -----------------

df = pd.concat([train, test])

print("Merged Shape:")
print(df.shape)

# -----------------
# Remove Duplicates
# -----------------

before = len(df)

df = df.drop_duplicates()

after = len(df)

print("Duplicates Removed:", before-after)

# -----------------
# Missing Values
# -----------------

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------
# Drop High Noise Columns
# -----------------

drop_cols = [
"id"
]

for c in drop_cols:
    if c in df.columns:
        df.drop(columns=c, inplace=True)

# -----------------
# Encode Categories
# -----------------

encoder = LabelEncoder()

for col in df.select_dtypes(include="object"):

    try:
        df[col] = encoder.fit_transform(
            df[col].astype(str)
        )

    except:
        pass

print("\nEncoded")

# -----------------
# Save Clean Data
# -----------------

df.to_csv(
"datasets/processed.csv",
index=False
)

print("\nSaved -> processed.csv")
print(df.head())