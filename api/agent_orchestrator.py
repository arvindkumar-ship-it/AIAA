"""
Unified Agent Orchestrator: Integrates all modules into a single autonomous agent.

Problem: Individual modules need orchestration, state management, and API integration.
Solution: Agent orchestrator that coordinates memory, intervention, style, autonomy, and reasoning.

Metrics:
- End-to-End Latency: Time from task creation to completion
- Task Success Rate: Fraction of tasks completed successfully
- User Satisfaction: Post-task rating (1-5)

Research Backing:
- Agentic Memory (arXiv:2601.01885): Unified memory management
- Intervention Timing (arXiv:2605.14830): Human-in-the-loop interventions
- Style Adaptation (ACL 2025): PATHs-based collaboration
- Autonomy Optimization (arXiv:2510.15739): AURA framework
- Reasoning & Planning (arXiv:2601.12538): Agentic reasoning
"""

from typing import Dict, List, Any, Optional, Tuple
from unittest import result
import numpy as np
from dataclasses import dataclass
import json
import os
import time
import structlog
from memory.memory_manager import MemoryManager
from intervention.intervention_predictor import InterventionPredictor, ActionFeature
from style.style_classifier import StyleClassifier, StyleAdapter, UserIntervention
from autonomy.autonomy_optimizer import AutonomyOptimizer, TaskRisk
from reasoning.planning_module import ExecutionEngine, ContextState

logger = structlog.get_logger()


@dataclass
class Task:
    """
    Task representation.
    """
    task_id: str
    task_type: str
    user_id: str
    input_data: Dict[str, Any]
    status: str = "pending"  # pending, executing, completed, failed
    output_data: Optional[Dict[str, Any]] = None
    intervention_required: bool = False
    intervention_payload: Optional[Dict[str, Any]] = None


class AgentOrchestrator:
    """
    Unified agent that orchestrates all modules.

    Usage:
        agent = AgentOrchestrator()
        result = agent.execute_task(task)
    """

    def __init__(self):
        # Initialize modules
        self.memory = MemoryManager()
        self.intervention_predictor = InterventionPredictor()
        self.style_classifier = StyleClassifier()
        self.autonomy_optimizer = AutonomyOptimizer()
        self.reasoning_engine = ExecutionEngine()

        # Task state
        self.task_history: List[Dict[str, Any]] = []

    def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute task end-to-end.

        Args:
            task: Task object

        Returns:
            Execution result
        """
        logger.info("Starting task execution", task_id=task.task_id, task_type=task.task_type)

        start_time = time.time()

        # Step 1: Get user style
        user_interventions = self._get_user_interventions(task.user_id)
        user_style, style_confidence = self.style_classifier.classify(user_interventions)
        style_adapter = StyleAdapter(user_style)

        logger.info("User style identified", user_id=task.user_id, style=user_style, confidence=style_confidence)

        # Step 2: Assess task risk
        task_risk = self._assess_task_risk(task)
        autonomy_level = self.autonomy_optimizer.get_autonomy_level(task_risk, user_style)
        autonomy_params = self.autonomy_optimizer.get_autonomy_params(autonomy_level)

        logger.info("Autonomy level determined", task_id=task.task_id, autonomy_level=autonomy_level)

        # Step 3: Store task in memory
        self.memory.store(
            f"Task {task.task_id}: {task.task_type} for user {task.user_id}",
            source="agent_action",
            task_id=task.task_id,
        )

        # Step 4: Execute task with intervention monitoring
        initial_context = ContextState(
            current_page="home",
            filled_fields={},
            pending_actions=[],
            completed_actions=[],
            errors=[],
        )

        success, final_context = self.reasoning_engine.execute(task.task_type, initial_context, task.input_data)

        # Monitor interventions during execution
        intervention_required = self._check_intervention_required(task, task_risk, autonomy_params)

        if intervention_required:
            # Ping user
            task.intervention_required = True
            task.intervention_payload = {
                "message": "Human approval required for this task",
                "autonomy_level": autonomy_level,
                "user_style": user_style,
            }
            logger.info("Intervention required", task_id=task.task_id)
            # In production: send webhook/email/SMS to user
            # Wait for user response (simplified)
            time.sleep(2)  # mock wait

        # Step 5: Update task status
        task.status = "completed" if success else "failed"
        task.output_data = {
            "completed_actions": final_context.completed_actions,
            "errors": final_context.errors,
            "user_style": user_style,
            "autonomy_level": autonomy_level,
        }

        # Step 6: Store result in memory
        self.memory.store(
            f"Task {task.task_id} completed: {task.status}",
            source="agent_action",
            task_id=task.task_id,
        )

        # Step 7: Log execution metrics
        execution_time = time.time() - start_time
        reasoning_metrics = self.reasoning_engine.get_metrics()
        memory_metrics = self.memory.get_metrics()

        logger.info(
            "Task execution completed",
            task_id=task.task_id,
            status=task.status,
            execution_time=f"{execution_time:.2f}s",
            success_rate=reasoning_metrics["success_rate"],
            memory_compression=memory_metrics["compression_ratio"],
        )

        # Step 8: Update task history
        self.task_history.append({
            "task_id": task.task_id,
            "task_type": task.task_type,
            "user_id": task.user_id,
            "status": task.status,
            "user_style": user_style,
            "autonomy_level": autonomy_level,
            "intervention_required": intervention_required,
            "execution_time": execution_time,
            "timestamp": time.time(),
        })

        return {
            "task_id": task.task_id,
            "status": task.status,
            "output_data": task.output_data,
            "intervention_required": intervention_required,
            "execution_time": execution_time,
        }

    def _get_user_interventions(self, user_id: str) -> List[UserIntervention]:
        """
        Get user's past interventions (from database or memory).

        In production, load from database.
        """
        # Simulated interventions (in production, load from DB)
        return [
            UserIntervention(task_type="bill_payment", action_index=4, intervention_type="approval", timestamp=time.time()),
            UserIntervention(task_type="appointment", action_index=5, intervention_type="correction", timestamp=time.time()),
        ]

    def _assess_task_risk(self, task: Task) -> TaskRisk:
        """
        Assess task risk.

        In production, use domain-specific risk models. For now, bill_payment's
        impact scales with the actual amount (normalized against ₹2000 as a
        "high impact" reference point) instead of a flat hardcoded value, so
        risk genuinely varies task-to-task instead of being constant for every
        task of a given type.
        """
        domain_map = {
            "bill_payment": "finance",
            "appointment": "healthcare",
            "paperwork": "legal",
        }

        if task.task_type == "bill_payment":
            amount = task.input_data.get("amount", 0)
            impact = min(float(amount) / 2000.0, 1.0) if isinstance(amount, (int, float)) else 0.5
        else:
            # No amount-like signal for these task types yet; keep low-impact
            # by default rather than the previous flat 0.5.
            impact = 0.2

        return TaskRisk(
            task_type=task.task_type,
            domain=domain_map.get(task.task_type, "general"),
            complexity=0.5,
            impact=impact,
            uncertainty=0.3,
        )

    def _check_intervention_required(self, task: Task, task_risk: TaskRisk, autonomy_params: Dict[str, Any]) -> bool:
        """
        Check if intervention is required.

        Decision hierarchy:
        1. Autonomy level's hard requirement (low autonomy => always ask).
        2. gated_decision() — combines the rule-based check (impact >= 0.3)
           with the learned model's probability, with the rule acting as a
           fallback/safety net until the model is trained on real/logged
           data (the LSTM in intervention_predictor.py is currently
           randomly initialized, not pre-trained).
        """
        # If autonomy level requires human approval
        if autonomy_params["human_approval_required"]:
            return True

        from gating.gating_logic import gated_decision

        gate_context = {
            "task_type": task.task_type,
            "amount": task.input_data.get("amount", 0),
            "deadline_hours": task.input_data.get("deadline_hours", 48),
            "is_recurring": int(task.input_data.get("is_recurring", False)),
            "time_of_day": task.input_data.get("time_of_day", time.localtime().tm_hour),
            "day_of_week": task.input_data.get("day_of_week", time.localtime().tm_wday),
        }

        result = gated_decision(gate_context)

        logger.info(
            "Intervention decision",
            task_id=task.task_id,
            decision_source=result["source"],
            model_prob=result["model_prob"],
            rule_decision=result["rule_decision"],
            final_decision=result["decision"],
        )

        return result["decision"]

    def get_user_style(self, user_id: str) -> Dict[str, Any]:
        """
        Get user's collaboration style.
        """
        user_interventions = self._get_user_interventions(user_id)
        style, confidence = self.style_classifier.classify(user_interventions)

        return {
            "user_id": user_id,
            "style": style,
            "confidence": confidence,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get agent performance metrics.
        """
        total_tasks = len(self.task_history)
        successful_tasks = sum(1 for t in self.task_history if t["status"] == "completed")
        avg_execution_time = np.mean([t["execution_time"] for t in self.task_history]) if self.task_history else 0.0

        reasoning_metrics = self.reasoning_engine.get_metrics()
        memory_metrics = self.memory.get_metrics()

        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0.0,
            "avg_execution_time": avg_execution_time,
            "reasoning_metrics": reasoning_metrics,
            "memory_metrics": memory_metrics,
        }