"""
Interactive real-time tester for AIAA.

Run: python demo/live_test.py
(Server must already be running — python run_server.py)

Type inputs, hit enter, see the full decision breakdown live:
risk score, intervention decision (rule vs model), success/fail reason.
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8001"  # change if your server runs on a different port

TASK_SCHEMAS = {
    "1": ("bill_payment", ["amount", "payee", "due"]),
    "2": ("appointment", ["title", "time"]),
    "3": ("paperwork", ["form_type"]),
}


def prompt_task():
    print("\nTask type:")
    print("  1) bill_payment  (fields: amount, payee, due)")
    print("  2) appointment   (fields: title, time)")
    print("  3) paperwork     (fields: form_type)")
    choice = input("Choose 1/2/3 (or 'q' to quit): ").strip()
    if choice.lower() == "q":
        sys.exit(0)
    if choice not in TASK_SCHEMAS:
        print("Invalid choice.")
        return None

    task_type, fields = TASK_SCHEMAS[choice]
    input_data = {}
    for field in fields:
        val = input(f"  {field}: ").strip()
        if field == "amount":
            try:
                val = float(val)
            except ValueError:
                print("  (not a number, sending as-is)")
        input_data[field] = val

    return task_type, input_data


def run_task(task_type, input_data):
    print(f"\n>>> Sending: {task_type} {input_data}")
    try:
        resp = requests.post(
            f"{BASE_URL}/tasks",
            json={"task_type": task_type, "user_id": "live_test_user", "input_data": input_data},
            timeout=15,
        )
    except requests.exceptions.ConnectionError:
        print(f"!! Can't reach {BASE_URL} — is the server running? (python run_server.py)")
        return

    if resp.status_code != 200:
        print(f"!! HTTP {resp.status_code}: {resp.text}")
        return

    result = resp.json()
    print("-" * 50)
    print(f"  Status:               {result['status']}")
    print(f"  Intervention required: {result['intervention_required']}")
    print(f"  Execution time:        {result['execution_time']:.2f}s")
    print(f"  Output data:")
    print(json.dumps(result["output_data"], indent=4))
    print("-" * 50)
    print("(Check the server terminal for the 'Intervention decision' log line —")
    print(" it shows rule_based_intervention, model_probability, and task_impact)")


def main():
    print("AIAA Live Tester — type tasks, see live decisions.")
    print(f"Talking to: {BASE_URL}")
    while True:
        task = prompt_task()
        if task:
            run_task(*task)


if __name__ == "__main__":
    main()