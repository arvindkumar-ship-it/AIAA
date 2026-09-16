"""
Test script for Intervention Predictor.

Run: python intervention/test_intervention.py

Expected Output:
- Intervention probability working
- AUC-ROC > 0.7
- PTS > 0.6
"""

import time
from intervention_predictor import InterventionPredictor, ActionFeature


def test_intervention():
    predictor = InterventionPredictor()
    
    # Test 1: Predict intervention probability
    print("Test 1: Predicting intervention probability...")
    action_sequence = [
        ActionFeature(
            action_type="navigate",
            action_data={"url": "bill_portal"},
            context={"task_type": "bill_payment", "complexity": 0.3, "risk_level": 0.2},
            user_history={"past_interventions": 5, "collaboration_style": "collaborative"},
        ),
        ActionFeature(
            action_type="fill_form",
            action_data={"amount": 900},
            context={"task_type": "bill_payment", "complexity": 0.5, "risk_level": 0.8},
            user_history={"past_interventions": 5, "collaboration_style": "collaborative"},
        ),
    ]
    
    start = time.time()
    prob = predictor.predict(action_sequence)
    latency = (time.time() - start) * 1000  # ms
    
    print(f"Intervention probability: {prob:.2f} (computed in {latency:.2f}ms)")
    
    # Test 2: Should intervene?
    print("Test 2: Should intervene?")
    should_ping = predictor.should_intervene(action_sequence, threshold=0.5)
    print(f"  Should ping: {should_ping}")
    
    # Test 3: Get metrics (dummy data)
    print("Test 3: Computing metrics...")
    predictions = [0.1, 0.4, 0.6, 0.8, 0.9]
    actuals = [False, False, True, True, True]
    metrics = predictor.get_metrics(predictions, actuals)
    
    print(f"  AUC-ROC: {metrics['auc_roc']:.2f}")
    print(f"  Brier Score: {metrics['brier_score']:.2f}")
    print(f"  PTS: {metrics['pts']:.2f}")
    
    # Assertions
    assert 0.0 <= prob <= 1.0, f"Invalid probability: {prob}"
    assert latency < 100, f"Prediction latency too high: {latency:.2f}ms"
    assert metrics['pts'] > 0.6, f"PTS too low: {metrics['pts']:.2f}"
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_intervention()
