# ==========================================================
# Frontend Layer: Ethical Workplace Copilot
# ==========================================================

import requests

API_URL = "http://127.0.0.1:5000/predict"


# ==========================================================
# 1️⃣ Call Backend API
# ==========================================================

def get_risk_prediction(employee_data):
    response = requests.post(API_URL, json=employee_data)
    
    if response.status_code != 200:
        raise Exception("Backend Error:", response.text)

    return response.json()


# ==========================================================
# 2️⃣ Generate Human-Friendly Explanation
# ==========================================================

def generate_copilot_message(employee_data, risk_output):

    risk_score = risk_output["risk_score"]
    risk_level = risk_output["risk_level"]
    signals = risk_output["signals"]

    message = "\n=====================================\n"
    message += "Workplace Wellbeing Copilot Report\n"
    message += "=====================================\n\n"

    message += f"Risk Level: {risk_level}\n"
    message += f"Risk Score (Next 30 Days): {risk_score}\n\n"

    if risk_level == "Low":
        message += (
            "Current work patterns appear stable.\n"
            "No immediate intervention required.\n"
            "Continue maintaining healthy workload balance.\n"
        )
        return message

    message += "Early Warning Indicators Detected:\n"

    for s in signals:
        message += f"• {s}\n"

    message += "\nPreventive Recommendations:\n"

    if "High weekly workload" in signals:
        message += "- Review workload distribution\n"

    if "Frequent absences" in signals:
        message += "- Check for potential disengagement or fatigue\n"

    if "Low job satisfaction" in signals:
        message += "- Schedule a one-on-one engagement discussion\n"

    if "Declining performance rating" in signals:
        message += "- Provide support or mentoring resources\n"

    message += "\nThis system is preventive and supportive.\n"
    message += "It is not intended for performance punishment.\n"

    return message


# ==========================================================
# 3️⃣ Example Usage
# ==========================================================

if __name__ == "__main__":

    employee_data = {
        "department": "IT",
        "role_level": 3,
        "monthly_salary": 55000,
        "avg_weekly_hours": 52,
        "projects_handled": 5,
        "performance_rating": 2,
        "absences_days": 10,
        "job_satisfaction": 2
    }

    risk_output = get_risk_prediction(employee_data)
    report = generate_copilot_message(employee_data, risk_output)

    print(report)