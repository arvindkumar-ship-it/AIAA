import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "logging"))
from db import export_training_data
from autonomy_optimizer import AutonomyOptimizer

logs = export_training_data()  # real logged tasks from Postgres
task_history = [
    {"task_type": r["task_id"], "autonomous": r["final_decision"] is False,
     "human_supervised": r["final_decision"] is True, "failed": r["outcome_success"] is False}
    for r in logs
]
optimizer = AutonomyOptimizer()
metrics = optimizer.compute_metrics(task_history)
print(f"Real N={metrics.total_tasks} | AIx={metrics.autonomy_index:.2f} | α={metrics.autonomy_coefficient:.2f}")