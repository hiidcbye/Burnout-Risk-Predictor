from flask import Flask, render_template, request
import requests

app = Flask(__name__)

API_URL = "http://127.0.0.1:5000/predict"

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        employee_data = {
            "department": request.form["department"],
            "role_level": int(request.form["role_level"]),
            "monthly_salary": float(request.form["monthly_salary"]),
            "avg_weekly_hours": float(request.form["avg_weekly_hours"]),
            "projects_handled": int(request.form["projects_handled"]),
            "performance_rating": int(request.form["performance_rating"]),
            "absences_days": int(request.form["absences_days"]),
            "job_satisfaction": int(request.form["job_satisfaction"])
        }

        response = requests.post(API_URL, json=employee_data)
        risk_output = response.json()

        return render_template(
            "index.html",
            show_result=True,
            form_data=employee_data,
            result=risk_output
        )

    return render_template("index.html", show_result=False)

if __name__ == "__main__":
    app.run(port=8000, debug=True)