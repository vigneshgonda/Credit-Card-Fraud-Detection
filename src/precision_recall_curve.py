import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score

df = pd.read_csv("data/processed/creditcard_balanced.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/best_fraud_model.pkl")

X_test_scaled = scaler.transform(X_test)

y_scores = model.predict_proba(X_test_scaled)[:, 1]

precision, recall, _ = precision_recall_curve(y_test, y_scores)
ap_score = average_precision_score(y_test, y_scores)

plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f"AP = {ap_score:.4f}", linewidth=2)

plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()

plt.savefig("outputs/figures/precision_recall_curve.png")
plt.show()

print("Precision-Recall Curve saved successfully!")