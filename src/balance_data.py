import pandas as pd
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("data/raw/creditcard.csv")

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

print("Original Class Distribution:")
print(y.value_counts())

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nAfter SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Create balanced dataframe
balanced_df = pd.concat(
    [
        pd.DataFrame(X_resampled, columns=X.columns),
        pd.Series(y_resampled, name="Class"),
    ],
    axis=1,
)

# Save balanced dataset
balanced_df.to_csv(
    "data/processed/creditcard_balanced.csv",
    index=False,
)

print("\nBalanced dataset saved successfully!")