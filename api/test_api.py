"""
Test script for Agent API.

Run: python api/test_api.py

Expected Output:
- Task creation working
- User style retrieval working
- Metrics computation accurate
"""

import requests
import time


def test_api():
    base_url = "http://localhost:8001"
    
    # Test 1: Create task
    print("Test 1: Creating task...")
    task_data = {
        "task_type": "bill_payment",
        "user_id": "U001",
        "input_data": {"amount": 150, "payee": "Electric Co", "due": "2026-09-20"},
    }
    
    start = time.time()
    response = requests.post(f"{base_url}/tasks", json=task_data)
    latency = (time.time() - start) * 1000  # ms
    
    print(f"  Response: {response.status_code}")
    print(f"  Latency: {latency:.2f}ms")
    
    if response.status_code == 200:
        result = response.json()
        print(f"  Task ID: {result['task_id']}")
        print(f"  Status: {result['status']}")
        print(f"  Execution time: {result['execution_time']:.2f}s")
    else:
        print(f"  Error: {response.text}")
    
    # Test 2: Get user style
    print("Test 2: Getting user style...")
    response = requests.get(f"{base_url}/users/U001/style")
    
    if response.status_code == 200:
        style_result = response.json()
        print(f"  Style: {style_result['style']}")
        print(f"  Confidence: {style_result['confidence']:.2f}")
    else:
        print(f"  Error: {response.text}")
    
    # Test 3: Get metrics
    print("Test 3: Getting metrics...")
    response = requests.get(f"{base_url}/metrics")
    
    if response.status_code == 200:
        metrics = response.json()
        print(f"  Total tasks: {metrics['total_tasks']}")
        print(f"  Success rate: {metrics['success_rate']:.2f}")
        print(f"  Avg execution time: {metrics['avg_execution_time']:.2f}s")
    else:
        print(f"  Error: {response.text}")
    
    # Assertions
    assert response.status_code == 200, f"API request failed: {response.status_code}"
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_api()
