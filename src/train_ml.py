import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# Load balanced dataset
df = pd.read_csv("data/processed/creditcard_balanced.csv")

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "Linear SVM": LinearSVC(random_state=42)
}

results = []
best_model = None
best_f1 = 0
best_model_name = ""

# Train and evaluate models
for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append([name, accuracy, precision, recall, f1])

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

# Results dataframe
results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
)

print("\nModel Comparison:\n")
print(results_df)

print(f"\nBest Model: {best_model_name}")
print(f"Best F1 Score: {best_f1}")

# Save best model and report
joblib.dump(best_model, "models/best_fraud_model.pkl")
results_df.to_csv("outputs/reports/model_comparison.csv", index=False)

print("\nBest model, scaler, and report saved successfully!")