# ==========================================================
# Backend ML Layer: Work Pattern Risk Engine
# Structured HR Dataset Version
# ==========================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ==========================================================
# 1️⃣ Dataset Generator
# ==========================================================

def generate_hr_dataset(n=500):

    departments = ["IT", "HR", "Finance", "Sales", "Operations"]

    df = pd.DataFrame({
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

    # Attrition Risk Logic (Burnout Proxy)
    risk_score = (
        0.04 * df["avg_weekly_hours"] +
        0.5 * df["absences_days"] -
        1.0 * df["job_satisfaction"] -
        0.02 * df["monthly_salary"] / 1000 +
        0.4 * (6 - df["performance_rating"])
    )

    prob = 1 / (1 + np.exp(-risk_score))
    df["attrition"] = np.where(np.random.rand(n) < prob, 1, 0)

    return df


# ==========================================================
# 2️⃣ Model Training Function
# ==========================================================

def train_model(df):

    X = df.drop(columns=["employee_id", "attrition"])
    y = df["attrition"]

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    print("Model Trained | ROC-AUC:", round(auc, 3))

    return model, X


# ==========================================================
# 3️⃣ Risk Engine Function
# ==========================================================

def generate_risk_output(model, X, employee_row):

    risk_score = model.predict_proba(pd.DataFrame([employee_row]))[0][1]

    # Risk Level
    if risk_score >= 0.75:
        risk_level = "High"
    elif risk_score >= 0.50:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    # Structured Rule-Based Signals
    signals = []

    if employee_row["avg_weekly_hours"] > 50:
        signals.append("High weekly workload")

    if employee_row["absences_days"] > 8:
        signals.append("Frequent absences")

    if employee_row["job_satisfaction"] <= 2:
        signals.append("Low job satisfaction")

    if employee_row["performance_rating"] <= 2:
        signals.append("Declining performance rating")

    return {
        "risk_score": round(float(risk_score), 3),
        "risk_level": risk_level,
        "signals": signals
    }


# ==========================================================
# 4️⃣ MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":

    df = generate_hr_dataset(500)

    print("Class Distribution:")
    print(df["attrition"].value_counts())

    model, X_full = train_model(df)

    print("\n=== Sample Backend Outputs ===")

    for i in range(5):
        employee_data = df.drop(columns=["attrition"]).iloc[i].to_dict()
        result = generate_risk_output(model, X_full, employee_data)
        print(result)