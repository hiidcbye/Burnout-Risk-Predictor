# ==========================================================
# Backend API - Work Pattern Risk Engine
# Flask Version
# ==========================================================

import joblib
import os

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
np.random.seed(42)

# ==========================================================
# 1️⃣ Generate Training Dataset
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
# 2️⃣ Train and save Model at Startup
# ==========================================================

MODEL_FILE = "model.pkl"

def train_and_save_model():
    df = generate_hr_dataset(500)

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

    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)
    print("Model trained and saved.")

    return model


# Load if exists, else train
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    print("Model loaded from file.")
else:
    model = train_and_save_model()

# ==========================================================
# 3️⃣ Risk Engine Logic
# ==========================================================

def generate_risk_output(employee_row):

    input_df = pd.DataFrame([employee_row])
    risk_score = model.predict_proba(input_df)[0][1]

    if risk_score >= 0.75:
        risk_level = "High"
    elif risk_score >= 0.50:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

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
# 4️⃣ API Endpoint
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    required_fields = [
        "department",
        "role_level",
        "monthly_salary",
        "avg_weekly_hours",
        "projects_handled",
        "performance_rating",
        "absences_days",
        "job_satisfaction"
    ]

    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    result = generate_risk_output(data)

    return jsonify(result)


# ==========================================================
# 5️⃣ Run Server
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True)