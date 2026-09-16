"""
Real test: feeds actual task logs (not 3 fake strings) through MemoryManager,
measures real compression on real content.
"""
import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "logging"))
from db import export_training_data
from memory_manager import MemoryManager

logs = export_training_data()
if not logs:
    print("No real logs yet — run some real tasks first (Calendar/orchestrator), then rerun this.")
    sys.exit()

memory = MemoryManager()
total_before, total_after = 0, 0

for r in logs:
    content = str(r["context"])  # real logged task context, not a fake sentence
    total_before += len(content)
    memory.store(content, source="agent_action", task_id=r["task_id"])
    total_after += len(memory.compressor.compress(content, target_ratio=0.3))

print(f"Real logged tasks used: {len(logs)}")
print(f"Total chars before: {total_before} | after: {total_after}")
print(f"Real compression ratio: {total_after/total_before:.2%}")

start = time.time()
results = memory.retrieve(str(logs[0]["context"]), top_k=3)
print(f"Retrieval latency: {(time.time()-start)*1000:.2f}ms | retrieved: {len(results)}")