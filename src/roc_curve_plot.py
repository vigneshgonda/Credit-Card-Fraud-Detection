import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Load balanced dataset
df = pd.read_csv("data/processed/creditcard_balanced.csv")

# Features and Target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load scaler and model
scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/best_fraud_model.pkl")

# Scale data
X_test_scaled = scaler.transform(X_test)

# Predict probabilities
y_scores = model.predict_proba(X_test_scaled)[:, 1]

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_scores)
auc = roc_auc_score(y_test, y_scores)

# Plot
plt.figure(figsize=(7, 5))

plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--")

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.savefig("outputs/figures/roc_curve.png")

plt.show()

print("ROC Curve saved successfully!")