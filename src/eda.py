import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/raw/creditcard.csv")

# -----------------------------
# Basic Information
# -----------------------------
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Class Distribution
# -----------------------------
print("\nFraud vs Normal Transactions:")
print(df["Class"].value_counts())

# -----------------------------
# Plot Class Distribution
# -----------------------------
plt.figure(figsize=(6,4))

sns.countplot(
    data=df,
    x="Class"
)

plt.title("Fraud vs Normal Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Count")

plt.savefig("outputs/figures/class_distribution.png")

plt.show()

# -----------------------------
# Statistics
# -----------------------------
print("\nOverall Statistics")
print(df.describe())

print("\nAmount Statistics")
print(df["Amount"].describe())

print("\nTime Statistics")
print(df["Time"].describe())

# -----------------------------
# Amount Distribution
# -----------------------------
plt.figure(figsize=(8,5))

sns.histplot(
    df["Amount"],
    bins=50
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")

plt.savefig("outputs/figures/amount_distribution.png")

plt.show()

print("\nEDA Completed Successfully!")