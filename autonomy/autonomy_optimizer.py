"""
Autonomy Optimizer: Risk-based autonomy adjustment using AURA framework.

Problem: Autonomous agents need to balance autonomy and safety in regulated contexts.
Solution: AURA framework (gamma-based risk scoring) + Autonomy Index (AIx) + AI Autonomy Coefficient (α).

Metrics:
- Autonomy Index (AIx): Fraction of consequential actions without human oversight
- AI Autonomy Coefficient (α): Proportion of tasks AI processes without mandatory human substitution
- Risk-Adjusted Autonomy Score: Weighted combination of autonomy and risk

Research Backing:
- AURA Framework (arXiv:2510.15739): Risk-based autonomy adjustment
- Autonomy Index (arXiv:2511.08242): Outcome-oriented evaluation
- AI Autonomy Coefficient (arXiv:2512.11295): Defining autonomy boundary
- Autonomy & Agency (arXiv:2605.12105): Architectural tactics for regulated contexts
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import json
import os
import time


@dataclass
class TaskRisk:
    """
    Task risk assessment.
    """
    task_type: str
    domain: str  # "healthcare", "finance", "legal", "general"
    complexity: float  # 0.0 to 1.0
    impact: float  # 0.0 to 1.0 (financial, health, legal impact)
    uncertainty: float  # 0.0 to 1.0 (model uncertainty)


@dataclass
class AutonomyMetrics:
    """
    Autonomy performance metrics.
    """
    total_tasks: int
    autonomous_tasks: int  # tasks completed without human oversight
    human_supervised_tasks: int  # tasks with human oversight
    failed_tasks: int  # tasks that failed
    autonomy_index: float  # AIx
    autonomy_coefficient: float  # α


class RiskScorer:
    """
    Computes task risk score using AURA framework's gamma-based scoring.
    
    Risk Score = w1 * complexity + w2 * impact + w3 * uncertainty
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "complexity": 0.4,
            "impact": 0.4,
            "uncertainty": 0.2,
        }
    
    def compute_risk(self, task_risk: TaskRisk) -> float:
        """
        Compute risk score (0.0 to 1.0).
        
        Args:
            task_risk: Task risk assessment
        
        Returns:
            Risk score
        """
        risk = (
            self.weights["complexity"] * task_risk.complexity +
            self.weights["impact"] * task_risk.impact +
            self.weights["uncertainty"] * task_risk.uncertainty
        )
        return min(max(risk, 0.0), 1.0)  # clamp to 0-1
    
    def get_domain_multiplier(self, domain: str) -> float:
        """
        Get domain-specific risk multiplier.
        
        Regulated domains (healthcare, finance, legal) have higher multipliers.
        """
        multipliers = {
            "healthcare": 1.5,
            "finance": 1.4,
            "legal": 1.4,
            "general": 1.0,
        }
        return multipliers.get(domain, 1.0)
    
    def compute_adjusted_risk(self, task_risk: TaskRisk) -> float:
        """
        Compute domain-adjusted risk score.
        
        Returns:
            Adjusted risk score (0.0 to 1.5)
        """
        base_risk = self.compute_risk(task_risk)
        multiplier = self.get_domain_multiplier(task_risk.domain)
        return base_risk * multiplier


class AutonomyOptimizer:
    """
    Optimizes agent autonomy level based on task risk and user style.
    
    Autonomy Levels:
    - low: Agent suggests actions, human approves each step
    - medium: Agent executes low-risk actions autonomously, high-risk require approval
    - high: Agent executes most actions autonomously, critical actions require approval
    - full: Agent executes all actions autonomously, human notified post-execution
    
    Usage:
        optimizer = AutonomyOptimizer()
        autonomy_level = optimizer.get_autonomy_level(task_risk, user_style)
    """
    
    AUTONOMY_LEVELS = ["low", "medium", "high", "full"]
    
    def __init__(self):
        self.risk_scorer = RiskScorer()
        self.autonomy_history: List[Dict[str, Any]] = []
    
    def get_autonomy_level(self, task_risk: TaskRisk, user_style: str) -> str:
        """
        Determine autonomy level based on task risk and user style.
        
        Args:
            task_risk: Task risk assessment
            user_style: User collaboration style
        
        Returns:
            Autonomy level ("low", "medium", "high", "full")
        """
        # Compute adjusted risk
        adjusted_risk = self.risk_scorer.compute_adjusted_risk(task_risk)
        
        # Base autonomy level from risk
        if adjusted_risk > 1.2:
            base_level = "low"
        elif adjusted_risk > 0.8:
            base_level = "medium"
        elif adjusted_risk > 0.4:
            base_level = "high"
        else:
            base_level = "full"
        
        # Adjust based on user style
        style_adjustments = {
            "takeover": -1,  # reduce autonomy by 1 level
            "hands_on": 0,  # no adjustment
            "hands_off": +1,  # increase autonomy by 1 level
            "collaborative": 0,  # no adjustment
        }
        
        adjustment = style_adjustments.get(user_style, 0)
        base_idx = self.AUTONOMY_LEVELS.index(base_level)
        adjusted_idx = max(0, min(len(self.AUTONOMY_LEVELS) - 1, base_idx + adjustment))
        
        autonomy_level = self.AUTONOMY_LEVELS[adjusted_idx]
        
        # Log decision
        self.autonomy_history.append({
            "task_type": task_risk.task_type,
            "adjusted_risk": adjusted_risk,
            "user_style": user_style,
            "autonomy_level": autonomy_level,
            "timestamp": time.time(),
        })
        
        return autonomy_level
    
    def get_autonomy_params(self, autonomy_level: str) -> Dict[str, Any]:
        """
        Get autonomy parameters for agent.
        
        Args:
            autonomy_level: Autonomy level
        
        Returns:
            Dictionary of parameters
        """
        params = {
            "low": {
                "human_approval_required": True,
                "explanation_required": True,
                "post_action_notification": False,
                "max_autonomous_actions": 0,
            },
            "medium": {
                "human_approval_required": False,
                "explanation_required": True,
                "post_action_notification": False,
                "max_autonomous_actions": 5,
            },
            "high": {
                "human_approval_required": False,
                "explanation_required": False,
                "post_action_notification": True,
                "max_autonomous_actions": 20,
            },
            "full": {
                "human_approval_required": False,
                "explanation_required": False,
                "post_action_notification": True,
                "max_autonomous_actions": -1,  # unlimited
            },
        }
        return params.get(autonomy_level, params["medium"])
    
    def compute_autonomy_index(self, total_tasks: int, autonomous_tasks: int) -> float:
        """
        Compute Autonomy Index (AIx).
        
        AIx = autonomous_tasks / total_tasks
        
        Args:
            total_tasks: Total tasks executed
            autonomous_tasks: Tasks completed without human oversight
        
        Returns:
            Autonomy Index (0.0 to 1.0)
        """
        if total_tasks == 0:
            return 0.0
        return autonomous_tasks / total_tasks
    
    def compute_autonomy_coefficient(self, total_tasks: int, successful_autonomous_tasks: int) -> float:
        """
        Compute AI Autonomy Coefficient (α).
        
        α = successful_autonomous_tasks / total_tasks
        
        Args:
            total_tasks: Total tasks executed
            successful_autonomous_tasks: Tasks AI processed successfully without human substitution
        
        Returns:
            Autonomy Coefficient (0.0 to 1.0)
        """
        if total_tasks == 0:
            return 0.0
        return successful_autonomous_tasks / total_tasks
    
    def compute_metrics(self, task_history: List[Dict[str, Any]]) -> AutonomyMetrics:
        """
        Compute autonomy metrics from task history.
        
        Args:
            task_history: List of task records with autonomy info
        
        Returns:
            AutonomyMetrics object
        """
        total_tasks = len(task_history)
        autonomous_tasks = sum(1 for t in task_history if t.get("autonomous", False))
        human_supervised_tasks = sum(1 for t in task_history if t.get("human_supervised", False))
        failed_tasks = sum(1 for t in task_history if t.get("failed", False))
        
        ai_index = self.compute_autonomy_index(total_tasks, autonomous_tasks)
        ai_coefficient = self.compute_autonomy_coefficient(total_tasks, autonomous_tasks)
        
        return AutonomyMetrics(
            total_tasks=total_tasks,
            autonomous_tasks=autonomous_tasks,
            human_supervised_tasks=human_supervised_tasks,
            failed_tasks=failed_tasks,
            autonomy_index=ai_index,
            autonomy_coefficient=ai_coefficient,
        )
