"""
Reasoning & Planning Module: Agentic reasoning with context compression and uncertainty handling.

Problem: LLM agents fail in long-horizon tasks due to planning weakness, context bloat, and uncertainty.
Solution: Agentic Reasoning Framework + Active Context Compression (Focus) + Dual-Level Uncertainty (WebUncertainty).

Metrics:
- Planning Efficiency: (optimal_steps / actual_steps) * 100
- Context Compression Ratio: (compressed_size / original_size) * 100
- Uncertainty Calibration: Brier score for uncertainty predictions

Research Backing:
- Agentic Reasoning (arXiv:2601.12538): Planning, tool use, search, self-correction
- Active Context Compression (arXiv:2601.07190): Autonomous context management
- WebUncertainty (ACL 2026): Dual-level uncertainty driven planning
- Memory as Action (arXiv:2510.12635): Working memory management for long-context LLMs
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
import json
import os
import time
from collections import deque
import heapq


@dataclass
class Action:
    """
    Atomic action in a plan.
    """
    action_type: str  # "navigate", "click", "type", "submit", "wait"
    action_data: Dict[str, Any]
    preconditions: List[str]  # required state before action
    postconditions: List[str]  # expected state after action
    uncertainty: float = 0.5  # 0.0 to 1.0 (action-level uncertainty)


@dataclass
class Plan:
    """
    Sequence of actions to achieve a goal.
    """
    goal: str
    actions: List[Action]
    total_uncertainty: float = 0.0  # trajectory-level uncertainty
    estimated_steps: int = 0


@dataclass
class ContextState:
    """
    Current context state for planning.
    """
    current_page: str
    filled_fields: Dict[str, Any]
    pending_actions: List[str]
    completed_actions: List[str]
    errors: List[str]


class UncertaintyEstimator:
    """
    Estimates action-level and trajectory-level uncertainty.
    
    Based on WebUncertainty's dual-level uncertainty framework.
    """
    
    def __init__(self):
        self.action_uncertainty_weights = {
            "navigate": 0.3,  # low uncertainty (deterministic)
            "click": 0.4,
            "type": 0.5,
            "submit": 0.6,
            "wait": 0.2,  # very low uncertainty
        }
    
    def estimate_action_uncertainty(self, action: Action) -> float:
        """
        Estimate action-level uncertainty.
        
        Factors:
        - Action type (navigate vs type)
        - Data complexity (simple text vs complex form)
        - Historical success rate
        """
        base_uncertainty = self.action_uncertainty_weights.get(action.action_type, 0.5)
        
        # Adjust based on data complexity
        data_complexity = len(str(action.action_data)) / 100.0  # normalize
        complexity_adjustment = min(data_complexity, 0.3)
        
        # Adjust based on preconditions
        precondition_penalty = len(action.preconditions) * 0.05
        
        uncertainty = base_uncertainty + complexity_adjustment + precondition_penalty
        return min(max(uncertainty, 0.0), 1.0)  # clamp to 0-1
    
    def estimate_trajectory_uncertainty(self, plan: Plan) -> float:
        """
        Estimate trajectory-level uncertainty (combined uncertainty of all actions).
        
        Formula: 1 - product(1 - action_uncertainty)
        """
        if not plan.actions:
            return 0.0
        
        product = 1.0
        for action in plan.actions:
            product *= (1.0 - action.uncertainty)
        
        trajectory_uncertainty = 1.0 - product
        return trajectory_uncertainty


class ContextCompressor:
    """
    Active context compression (Focus framework).
    
    Agent autonomously decides when to consolidate vs prune context.
    """
    
    def __init__(self, max_context_length: int = 10000):
        self.max_context_length = max_context_length
        self.compression_ratio = 0.0
    
    def should_compress(self, context: str) -> bool:
        """
        Decide whether to compress context.
        
        Criteria:
        - Context length > max_context_length
        - Redundant information detected
        """
        if len(context) > self.max_context_length:
            return True
        
        # Check for redundancy (simplified: repeated phrases)
        words = context.split()
        unique_words = set(words)
        redundancy = 1.0 - (len(unique_words) / len(words)) if words else 0.0
        
        return redundancy > 0.5
    
    def compress(self, context: str, mode: str = "prune") -> str:
        """
        Compress context.
        
        Modes:
        - prune: Remove redundant information
        - summarize: Generate summary (requires LLM)
        - consolidate: Merge related information
        """
        if mode == "prune":
            compressed = self._prune_redundancy(context)
        elif mode == "summarize":
            compressed = self._summarize(context)  # TODO: implement with LLM
        elif mode == "consolidate":
            compressed = self._consolidate(context)
        else:
            compressed = context
        
        self.compression_ratio = len(compressed) / len(context) if context else 1.0
        return compressed
    
    def _prune_redundancy(self, context: str) -> str:
        """
        Remove redundant information from context.
        """
        sentences = context.split(". ")
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            if sentence not in seen:
                unique_sentences.append(sentence)
                seen.add(sentence)
        
        return ". ".join(unique_sentences)
    
    def _summarize(self, context: str) -> str:
        """
        Generate summary of context.
        
        In production, use LLM for summarization.
        """
        # Simplified: return first 3 sentences
        sentences = context.split(". ")
        return ". ".join(sentences[:3])
    
    def _consolidate(self, context: str) -> str:
        """
        Consolidate related information.
        """
        # Simplified: group by topic (keyword-based)
        sentences = context.split(". ")
        topics = {}
        
        for sentence in sentences:
            # Extract topic (first word as proxy)
            words = sentence.split()
            if words:
                topic = words[0].lower()
                topics[topic] = topics.get(topic, []) + [sentence]
        
        # Consolidate
        consolidated = []
        for topic, topic_sentences in topics.items():
            consolidated.append(f"[{topic}] " + ". ".join(topic_sentences))
        
        return "\n".join(consolidated)


class Planner:
    """
    Generates plans for achieving goals.
    
    Uses BFS/DFS for simple planning, LLM for complex planning.
    """
    
    def __init__(self):
        self.action_templates = self._load_action_templates()
    
    def _load_action_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Load action templates for common tasks.
        """
        return {
            "bill_payment": {
                "actions": [
                    Action("navigate", {"url": "bill_portal"}, [], ["on_bill_portal"]),
                    Action("fill_form", {"fields": ["amount", "account"]}, ["on_bill_portal"], ["form_filled"]),
                    Action("submit", {}, ["form_filled"], ["payment_submitted"]),
                    Action("wait", {"seconds": 2}, ["payment_submitted"], ["payment_confirmed"]),
                ],
                "estimated_steps": 4,
            },
            "appointment": {
                "actions": [
                    Action("navigate", {"url": "calendar"}, [], ["on_calendar"]),
                    Action("click", {"element": "time_slot"}, ["on_calendar"], ["slot_selected"]),
                    Action("fill_form", {"fields": ["name", "email"]}, ["slot_selected"], ["form_filled"]),
                    Action("submit", {}, ["form_filled"], ["appointment_booked"]),
                ],
                "estimated_steps": 4,
            },
            "paperwork": {
                "actions": [
                    Action("navigate", {"url": "form_portal"}, [], ["on_form_portal"]),
                    Action("fill_form", {"fields": ["name", "address", "tin"]}, ["on_form_portal"], ["form_filled"]),
                    Action("submit", {}, ["form_filled"], ["form_submitted"]),
                    Action("wait", {"seconds": 3}, ["form_submitted"], ["form_confirmed"]),
                ],
                "estimated_steps": 4,
            },
        }
    
    def generate_plan(self, goal: str, context: ContextState) -> Plan:
        """
        Generate plan for achieving goal.
        
        Args:
            goal: Goal description (e.g., "pay_bill", "book_appointment")
            context: Current context state
        
        Returns:
            Plan object
        """
        # Get action template
        template = self.action_templates.get(goal, self.action_templates["bill_payment"])
        
        # Create plan
        actions = template["actions"].copy()
        
        # Update actions based on context
        for action in actions:
            # Update preconditions based on completed actions
            action.preconditions = [
                p for p in action.preconditions
                if p not in context.completed_actions
            ]
        
        # Estimate uncertainties
        uncertainty_estimator = UncertaintyEstimator()
        for action in actions:
            action.uncertainty = uncertainty_estimator.estimate_action_uncertainty(action)
        
        total_uncertainty = uncertainty_estimator.estimate_trajectory_uncertainty(
            Plan(goal=goal, actions=actions)
        )
        
        return Plan(
            goal=goal,
            actions=actions,
            total_uncertainty=total_uncertainty,
            estimated_steps=template["estimated_steps"],
        )
    
    def replan(self, plan: Plan, error: str, context: ContextState) -> Plan:
        """
        Replan after error.
        
        Args:
            plan: Original plan
            error: Error description
            context: Current context state
        
        Returns:
            New plan
        """
        # Simplified replanning (in production, use LLM)
        # Find failed action
        failed_action_idx = len(context.completed_actions)
        
        if failed_action_idx >= len(plan.actions):
            # All actions completed, but goal not achieved → retry
            return self.generate_plan(plan.goal, context)
        
        # Skip failed action, continue from next
        new_actions = plan.actions[failed_action_idx + 1:]
        
        total_uncertainty = UncertaintyEstimator().estimate_trajectory_uncertainty(
            Plan(goal=plan.goal, actions=new_actions)
        )
        
        return Plan(
            goal=plan.goal,
            actions=new_actions,
            total_uncertainty=total_uncertainty,
            estimated_steps=len(new_actions),
        )


class ExecutionEngine:
    """
    Executes plans with uncertainty monitoring and context compression.
    """
    
    # Fields each task type actually needs to be considered valid/complete.
    # These match the real input_data shape the API receives (see api/routes.py
    # TaskCreate / the demo task list), not the older "account"-style template
    # field names that never appeared in any real request.
    REQUIRED_FIELDS = {
        "bill_payment": ["amount", "payee"],
        "appointment": ["title", "time"],
        "paperwork": ["form_type"],
    }
    
    def __init__(self):
        self.planner = Planner()
        self.context_compressor = ContextCompressor()
        self.execution_history: List[Dict[str, Any]] = []
    
    def execute(self, goal: str, initial_context: ContextState, task_data: Optional[Dict[str, Any]] = None) -> Tuple[bool, ContextState]:
        """
        Execute plan for goal.
        
        Args:
            goal: Goal description
            initial_context: Initial context state
            task_data: The actual task input data (e.g. {"amount": 900, "payee": "..."}).
                       Carried into context.filled_fields and used to decide whether
                       data-dependent actions (fill_form/submit) actually succeed.
        
        Returns:
            (success, final_context)
        """
        task_data = task_data or {}
        
        # Generate plan
        plan = self.planner.generate_plan(goal, initial_context)
        context = initial_context
        context.filled_fields = dict(task_data)
        
        is_valid, missing_fields = self._validate_task_data(goal, task_data)
        
        # Execute actions
        for i, action in enumerate(plan.actions):
            # Check preconditions
            if not self._check_preconditions(action, context):
                # Replan
                plan = self.planner.replan(plan, f"Precondition failed: {action.preconditions}", context)
            
            # Execute action
            success = self._execute_action(action, context, is_valid)
            
            if not success:
                context.errors.append(f"{action.action_type} failed: missing/invalid fields {missing_fields}")
                # Replan
                plan = self.planner.replan(plan, f"Action failed: {action.action_type}", context)
                # Retry
                success = self._execute_action(action, context, is_valid)
                if not success:
                    return False, context
            
            # Update context
            context.completed_actions.append(action.action_type)
            for postcondition in action.postconditions:
                if postcondition not in context.pending_actions:
                    context.pending_actions.append(postcondition)
            
            # Check for context compression
            context_str = json.dumps(context.__dict__)
            if self.context_compressor.should_compress(context_str):
                compressed = self.context_compressor.compress(context_str, mode="prune")
                context = ContextState(**json.loads(compressed))
            
            # Log execution
            self.execution_history.append({
                "action": action.action_type,
                "success": success,
                "uncertainty": action.uncertainty,
                "timestamp": time.time(),
            })
        
        return True, context
    
    def _validate_task_data(self, goal: str, task_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that the task's actual input data is complete/sane for its type.
        This is what real success/failure is now based on, instead of a random roll.
        """
        required = self.REQUIRED_FIELDS.get(goal, [])
        missing = [f for f in required if task_data.get(f) in (None, "", 0)]
        
        if goal == "bill_payment":
            amount = task_data.get("amount")
            if not isinstance(amount, (int, float)) or amount <= 0:
                if "amount" not in missing:
                    missing.append("amount")
        
        return (len(missing) == 0, missing)
    
    def _check_preconditions(self, action: Action, context: ContextState) -> bool:
        """
        Check if action preconditions are satisfied.
        """
        for precondition in action.preconditions:
            if precondition == "on_bill_portal" and context.current_page != "bill_portal":
                return False
            if precondition == "form_filled" and not context.filled_fields:
                return False
        return True
    
    def _execute_action(self, action: Action, context: ContextState, is_valid: bool) -> bool:
        """
        Execute action.
        
        Deterministic and data-driven (previously: random 90% coin flip regardless
        of the task's actual data). There is no real external portal for
        navigate/wait to actually reach or fail against, so those steps succeed
        unconditionally. fill_form/submit/click depend on whether the task's
        real input data was complete and valid (see _validate_task_data) — so
        the same task with the same data always produces the same result, and a
        genuinely incomplete/invalid task (e.g. amount <= 0, missing payee)
        will genuinely fail instead of passing by chance.
        """
        if action.action_type in ("navigate", "wait"):
            return True
        return is_valid
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get execution metrics.
        """
        total_actions = len(self.execution_history)
        successful_actions = sum(1 for h in self.execution_history if h["success"])
        avg_uncertainty = np.mean([h["uncertainty"] for h in self.execution_history]) if self.execution_history else 0.0
        
        return {
            "total_actions": total_actions,
            "successful_actions": successful_actions,
            "success_rate": successful_actions / total_actions if total_actions > 0 else 0.0,
            "avg_uncertainty": avg_uncertainty,
            "context_compression_ratio": self.context_compressor.compression_ratio,
        }