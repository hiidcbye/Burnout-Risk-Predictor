import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "department": "IT",
    "role_level": 3,
    "monthly_salary": 55000,
    "avg_weekly_hours": 52,
    "projects_handled": 5,
    "performance_rating": 2,
    "absences_days": 10,
    "job_satisfaction": 2
}

response = requests.post(url, json=data)
print(response.json())