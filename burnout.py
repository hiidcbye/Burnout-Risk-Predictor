# ==========================================================
# HR Attrition Risk Prediction + Ethical Workplace Copilot
# Single-File Structured Dataset Version

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ==========================================================
#  Generate Structured HR Dataset

def generate_hr_dataset(n=500):

    departments = ["IT", "HR", "Finance", "Sales", "Operations"]
    
    data = pd.DataFrame({
        "employee_id": np.arange(n),
        "department": np.random.choice(departments, n),
        "role_level": np.random.randint(1, 6, n),
        "monthly_salary": np.random.normal(60000, 15000, n),
        "avg_weekly_hours": np.random.normal(45, 5, n),
        "projects_handled": np.random.randint(1, 10, n),
        "performance_rating": np.random.randint(1, 6, n),
        "absences_days": np.random.randint(0, 15, n),
        "job_satisfaction": np.random.randint(1, 6, n)
    })

    # Create realistic attrition probability
    risk_score = (
        0.03 * data["avg_weekly_hours"] +
        0.4 * data["absences_days"] -
        0.8 * data["job_satisfaction"] -
        0.02 * data["monthly_salary"] / 1000 +
        0.3 * (6 - data["performance_rating"])
    )

    prob = 1 / (1 + np.exp(-risk_score))
    data["attrition"] = np.where(np.random.rand(n) < prob, 1, 0)

    return data


df = generate_hr_dataset(500)

print("Dataset Shape:", df.shape)
print("\nClass Distribution:")
print(df["attrition"].value_counts())


# ==========================================================
#  Train/Test Split

X = df.drop(columns=["employee_id", "attrition"])
y = df["attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==========================================================
#  Preprocessing + Model Pipeline

categorical = ["department"]
numeric = [
    "role_level",
    "monthly_salary",
    "avg_weekly_hours",
    "projects_handled",
    "performance_rating",
    "absences_days",
    "job_satisfaction"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical),
        ("num", "passthrough", numeric)
    ]
)

model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000))
])

model.fit(X_train, y_train)


# ==========================================================
# Model Evaluation

probs = model.predict_proba(X_test)[:, 1]
preds = (probs > 0.5).astype(int)

print("\nModel Performance")
print("ROC-AUC:", round(roc_auc_score(y_test, probs), 3))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))
print("\nClassification Report:")
print(classification_report(y_test, preds))


# ==========================================================
# Risk Scoring for Entire Dataset

df["risk_score"] = model.predict_proba(X)[:, 1]


# ==========================================================
# Ethical Workplace Copilot Layer

def wellbeing_copilot(row):

    explanation = "\n-------------------------------------\n"
    explanation += f"Employee ID: {row['employee_id']}\n"
    explanation += f"Attrition Risk Score (Burnout Proxy): {round(row['risk_score'], 2)}\n"

    signals = []

    if row["avg_weekly_hours"] > 50:
        signals.append("High weekly working hours")

    if row["absences_days"] > 8:
        signals.append("Frequent absences")

    if row["job_satisfaction"] <= 2:
        signals.append("Low job satisfaction")

    if row["performance_rating"] <= 2:
        signals.append("Declining performance rating")

    if len(signals) == 0:
        explanation += "No major risk indicators detected.\n"
        explanation += "Continue maintaining current work patterns.\n"
        return explanation

    explanation += "\nDetected Risk Signals:\n"
    for s in signals:
        explanation += f"• {s}\n"

    explanation += "\nPreventive Recommendations:\n"
    explanation += "- Review workload balance\n"
    explanation += "- Conduct engagement discussion\n"
    explanation += "- Consider flexible scheduling\n"
    explanation += "- Provide wellbeing support resources\n"

    return explanation


# ==========================================================
# Sample Copilot Outputs

print("\n\n=== Copilot Output Samples ===")

for i in range(5):
    print(wellbeing_copilot(df.iloc[i]))

