"""
Test script for Reasoning & Planning Module.

Run: python reasoning/test_reasoning.py

Expected Output:
- Plan generation working
- Context compression working
- Uncertainty estimation accurate
- Execution success rate > 0.8
"""

import time
from planning_module import ExecutionEngine, ContextState


def test_reasoning():
    engine = ExecutionEngine()
    
    # Test 1: Generate and execute plan
    print("Test 1: Executing plan for bill_payment...")
    initial_context = ContextState(
        current_page="home",
        filled_fields={},
        pending_actions=[],
        completed_actions=[],
        errors=[],
    )
    
    start = time.time()
    success, final_context = engine.execute(
        "bill_payment",
        initial_context,
        task_data={"amount": 150, "payee": "Electric Co", "due": "2026-09-20"},
    )
    latency = (time.time() - start) * 1000  # ms
    
    print(f"  Success: {success} (computed in {latency:.2f}ms)")
    print(f"  Completed actions: {final_context.completed_actions}")
    
    # Test 2: Get execution metrics
    print("Test 2: Getting execution metrics...")
    metrics = engine.get_metrics()
    
    print(f"  Total actions: {metrics['total_actions']}")
    print(f"  Successful actions: {metrics['successful_actions']}")
    print(f"  Success rate: {metrics['success_rate']:.2f}")
    print(f"  Avg uncertainty: {metrics['avg_uncertainty']:.2f}")
    print(f"  Context compression ratio: {metrics['context_compression_ratio']:.2f}")
    
    # Test 3: Context compression
    print("Test 3: Testing context compression...")
    long_context = " ".join(["This is a test sentence. "] * 100)
    should_compress = engine.context_compressor.should_compress(long_context)
    compressed = engine.context_compressor.compress(long_context, mode="prune")
    
    print(f"  Should compress: {should_compress}")
    print(f"  Original length: {len(long_context)}")
    print(f"  Compressed length: {len(compressed)}")
    print(f"  Compression ratio: {len(compressed) / len(long_context):.2f}")
    
    # Assertions
    assert latency < 1000, f"Execution latency too high: {latency:.2f}ms"
    assert metrics['success_rate'] > 0.8, f"Success rate too low: {metrics['success_rate']:.2f}"
    assert len(compressed) < len(long_context), "Compression failed"
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_reasoning()