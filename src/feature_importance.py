import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("models/best_fraud_model.pkl")

# Load dataset
df = pd.read_csv("data/processed/creditcard_balanced.csv")

X = df.drop("Class", axis=1)

# Get feature importance from Random Forest
importance = model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print(feature_df.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(
    data=feature_df.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.savefig("outputs/figures/feature_importance.png")
plt.show()

print("Feature importance saved successfully!")