"""
Run 10 real demo tasks and record metrics.
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8001"

tasks = [
    {"task_type": "bill_payment", "input_data": {"amount": 150, "payee": "Electric Co", "due": "2026-09-20"}},
    {"task_type": "bill_payment", "input_data": {"amount": 900, "payee": "Electric Co", "due": "2026-09-20"}},
    {"task_type": "appointment", "input_data": {"title": "Dentist", "time": "10:00 AM"}},
    {"task_type": "appointment", "input_data": {"title": "Doctor", "time": "3:00 PM"}},
    {"task_type": "paperwork", "input_data": {"form_type": "Tax W-8BEN"}},
    {"task_type": "bill_payment", "input_data": {"amount": 50, "payee": "Internet Co", "due": "2026-09-25"}},
    {"task_type": "appointment", "input_data": {"title": "Gym", "time": "6:00 PM"}},
    {"task_type": "bill_payment", "input_data": {"amount": 1200, "payee": "Credit Card", "due": "2026-09-22"}},
    {"task_type": "paperwork", "input_data": {"form_type": "Passport Form"}},
    {"task_type": "appointment", "input_data": {"title": "Salon", "time": "2:00 PM"}},
]

results = []

print("Running 10 demo tasks...\n")

for i, task in enumerate(tasks, 1):
    print(f"Task {i}/10: {task['task_type']} - {task['input_data']}")
    
    response = requests.post(f"{BASE_URL}/tasks", json={
        "task_type": task["task_type"],
        "user_id": "demo_user",
        "input_data": task["input_data"],
    })
    
    result = response.json()
    results.append({
        "task_num": i,
        "task_type": task["task_type"],
        "input": task["input_data"],
        "status": result["status"],
        "intervention_required": result["intervention_required"],
        "execution_time": result["execution_time"],
    })
    
    print(f"  Status: {result['status']}")
    print(f"  Intervention: {result['intervention_required']}")
    print(f"  Time: {result['execution_time']:.2f}s\n")
    
    time.sleep(1)  # thoda delay between tasks

# Save results
with open("demo/demo_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Calculate metrics
total_tasks = len(results)
successful = sum(1 for r in results if r["status"] == "completed")
interventions = sum(1 for r in results if r["intervention_required"])
avg_time = sum(r["execution_time"] for r in results) / total_tasks

metrics = {
    "total_tasks": total_tasks,
    "successful_tasks": successful,
    "success_rate": successful / total_tasks,
    "interventions": interventions,
    "intervention_rate": interventions / total_tasks,
    "avg_execution_time": avg_time,
    "timestamp": datetime.now().isoformat(),
}

print("=" * 50)
print("DEMO METRICS:")
print(f"  Total Tasks: {total_tasks}")
print(f"  Successful: {successful} ({metrics['success_rate']*100:.1f}%)")
print(f"  Interventions: {interventions} ({metrics['intervention_rate']*100:.1f}%)")
print(f"  Avg Execution Time: {avg_time:.2f}s")
print("=" * 50)

with open("demo/demo_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nResults saved to demo/demo_results.json")
print("Metrics saved to demo/demo_metrics.json")