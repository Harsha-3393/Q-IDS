import pandas as pd

train = pd.read_csv("datasets/UNSW_NB15_training-set.csv")
test = pd.read_csv("datasets/UNSW_NB15_testing-set.csv")

print("\nTraining Shape:")
print(train.shape)

print("\nTesting Shape:")
print(test.shape)

print("\nColumns:")
print(train.columns)

print("\nPreview:")
print(train.head())