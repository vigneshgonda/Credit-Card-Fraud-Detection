import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/creditcard.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset Info:")
print(df.info())