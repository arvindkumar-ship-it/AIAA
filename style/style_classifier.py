"""
Style Classifier: Identifies user collaboration styles and adapts agent behavior.

Problem: Users have different collaboration preferences (hands_on vs hands_off).
Solution: PATHs-based feature extraction + Random Forest classification + real-time adaptation.

Metrics:
- Classification Accuracy: Fraction of correct style predictions
- Adaptation Score: Improvement in user satisfaction after adaptation
- Style Stability: Consistency of style over time

Research Backing:
- PATHs (ACL 2025): Prototypical Human-AI Collaboration Behaviors
- 5-Level Taxonomy (arXiv:2606.15509): Diagnostic framework for collaboration
- CowCorpus (arXiv:2602.17588): 4 distinct collaboration styles
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time


@dataclass
class UserIntervention:
    """
    User intervention record.
    """
    task_type: str
    action_index: int
    intervention_type: str  # "takeover", "correction", "approval", "skip"
    timestamp: float


@dataclass
class UserProfile:
    """
    User profile with style and history.
    """
    user_id: str
    style: Optional[str]  # "takeover", "hands_on", "hands_off", "collaborative"
    interventions: List[UserIntervention]
    style_confidence: float  # 0.0 to 1.0


class FeatureExtractor:
    """
    Extracts features from user intervention history for style classification.
    
    Based on PATHs (ACL 2025) and CowCorpus feature design.
    """
    
    def __init__(self):
        self.feature_names = [
            "total_interventions",
            "avg_intervention_index",
            "takeover_ratio",
            "correction_ratio",
            "approval_ratio",
            "skip_ratio",
            "task_type_diversity",
            "intervention_frequency",
            "early_intervention_ratio",
            "late_intervention_ratio",
        ]
    
    def extract(self, interventions: List[UserIntervention]) -> np.ndarray:
        """
        Extract features from intervention history.
        
        Args:
            interventions: List of user interventions
        
        Returns:
            Feature vector (10-dimensional)
        """
        if not interventions:
            # Default features for new users (neutral style)
            return np.array([0.0, 0.5, 0.25, 0.25, 0.25, 0.25, 0.0, 0.5, 0.5, 0.5])
        
        total = len(interventions)
        
        # Feature 1: Total interventions (normalized)
        total_interventions = min(total / 100.0, 1.0)
        
        # Feature 2: Average intervention index (early vs late)
        avg_index = np.mean([i.action_index for i in interventions]) / 10.0  # normalize to 0-1
        
        # Features 3-6: Intervention type distribution
        type_counts = {}
        for i in interventions:
            type_counts[i.intervention_type] = type_counts.get(i.intervention_type, 0) + 1
        takeover_ratio = type_counts.get("takeover", 0) / total
        correction_ratio = type_counts.get("correction", 0) / total
        approval_ratio = type_counts.get("approval", 0) / total
        skip_ratio = type_counts.get("skip", 0) / total
        
        # Feature 7: Task type diversity
        task_types = set(i.task_type for i in interventions)
        task_diversity = len(task_types) / 5.0  # normalize (assume max 5 task types)
        
        # Feature 8: Intervention frequency (interventions per task)
        task_counts = {}
        for i in interventions:
            task_counts[i.task_type] = task_counts.get(i.task_type, 0) + 1
        avg_interventions_per_task = np.mean(list(task_counts.values()))
        intervention_frequency = min(avg_interventions_per_task / 5.0, 1.0)
        
        # Features 9-10: Early vs late intervention ratio
        early_threshold = 3  # action_index < 3 is "early"
        early_interventions = sum(1 for i in interventions if i.action_index < early_threshold)
        early_ratio = early_interventions / total
        late_ratio = 1.0 - early_ratio
        
        features = np.array([
            total_interventions,
            avg_index,
            takeover_ratio,
            correction_ratio,
            approval_ratio,
            skip_ratio,
            task_diversity,
            intervention_frequency,
            early_ratio,
            late_ratio,
        ])
        
        return features


class StyleClassifier:
    """
    Classifies user collaboration style based on intervention history.
    
    Styles:
    - takeover: User late me intervene karta hai aur control wapas nahi karta
    - hands_on: User frequently intervene karta hai
    - hands_off: User rarely intervene karta hai
    - collaborative: User selectively intervene karta hai
    
    Usage:
        classifier = StyleClassifier()
        style = classifier.classify(user_interventions)
        adapter = StyleAdapter(style)
        threshold = adapter.get_intervention_threshold()
    """
    
    STYLES = ["takeover", "hands_on", "hands_off", "collaborative"]
    
    def __init__(self, model_path: str = "./style/style_classifier.pkl"):
        self.model_path = model_path
        self.feature_extractor = FeatureExtractor()
        self.model = self._load_or_train()
    
    def _load_or_train(self):
        """
        Load pre-trained model or train new one.
        """
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        else:
            # Train on simulated CowCorpus data
            return self._train_on_simulated_data()
    
    def _train_on_simulated_data(self):
        """
        Train classifier on simulated CowCorpus-like data.
        In production, use real CowCorpus dataset.
        """
        # Simulated training data (in production, load from CowCorpus)
        np.random.seed(42)
        n_samples = 200
        
        # Generate synthetic intervention histories for each style
        X = []
        y = []
        
        for style_idx, style in enumerate(self.STYLES):
            for _ in range(n_samples // len(self.STYLES)):
                # Generate intervention history based on style
                interventions = self._generate_simulated_interventions(style)
                features = self.feature_extractor.extract(interventions)
                X.append(features)
                y.append(style_idx)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train Random Forest
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight="balanced")
        model.fit(X, y)
        
        # Save model
        os.makedirs("style", exist_ok=True)
        joblib.dump(model, self.model_path)
        
        return model
    
    def _generate_simulated_interventions(self, style: str) -> List[UserIntervention]:
        """
        Generate simulated intervention history for a given style.
        """
        interventions = []
        
        if style == "takeover":
            # Few interventions, mostly late, high takeover ratio
            num_interventions = np.random.randint(2, 5)
            for _ in range(num_interventions):
                interventions.append(UserIntervention(
                    task_type=np.random.choice(["bill_payment", "appointment", "paperwork"]),
                    action_index=np.random.randint(5, 10),  # late
                    intervention_type="takeover" if np.random.random() > 0.3 else "correction",
                    timestamp=time.time(),
                ))
        
        elif style == "hands_on":
            # Many interventions, distributed, high correction ratio
            num_interventions = np.random.randint(10, 20)
            for _ in range(num_interventions):
                interventions.append(UserIntervention(
                    task_type=np.random.choice(["bill_payment", "appointment", "paperwork"]),
                    action_index=np.random.randint(1, 8),
                    intervention_type="correction" if np.random.random() > 0.5 else "approval",
                    timestamp=time.time(),
                ))
        
        elif style == "hands_off":
            # Very few interventions, mostly approvals
            num_interventions = np.random.randint(1, 3)
            for _ in range(num_interventions):
                interventions.append(UserIntervention(
                    task_type=np.random.choice(["bill_payment", "appointment", "paperwork"]),
                    action_index=np.random.randint(3, 6),
                    intervention_type="approval",
                    timestamp=time.time(),
                ))
        
        elif style == "collaborative":
            # Moderate interventions, selective, mixed types
            num_interventions = np.random.randint(5, 10)
            for _ in range(num_interventions):
                interventions.append(UserIntervention(
                    task_type=np.random.choice(["bill_payment", "appointment", "paperwork"]),
                    action_index=np.random.randint(3, 7),
                    intervention_type=np.random.choice(["approval", "correction", "skip"]),
                    timestamp=time.time(),
                ))
        
        return interventions
    
    def classify(self, interventions: List[UserIntervention]) -> Tuple[str, float]:
        """
        Classify user's collaboration style.
        
        Args:
            interventions: List of user interventions
        
        Returns:
            (style, confidence)
        """
        features = self.feature_extractor.extract(interventions)
        features = features.reshape(1, -1)
        
        # Predict style
        style_idx = self.model.predict(features)[0]
        style = self.STYLES[style_idx]
        
        # Get confidence (probability)
        proba = self.model.predict_proba(features)[0]
        confidence = proba[style_idx]
        
        return style, confidence
    
    def get_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary of metrics
        """
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.STYLES, output_dict=True)
        
        return {
            "accuracy": accuracy,
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "per_class_f1": {style: report[style]["f1-score"] for style in self.STYLES},
        }


class StyleAdapter:
    """
    Adapts agent behavior based on user style.
    
    Usage:
        adapter = StyleAdapter("collaborative")
        threshold = adapter.get_intervention_threshold()
        should_ping = prob > threshold
    """
    
    def __init__(self, style: str):
        self.style = style
    
    def get_intervention_threshold(self) -> float:
        """
        Get intervention threshold based on style.
        
        Returns:
            Threshold (0.0 to 1.0)
        """
        thresholds = {
            "takeover": 0.8,  # only ping if very high probability
            "hands_on": 0.4,  # ping frequently
            "hands_off": 0.9,  # ping rarely
            "collaborative": 0.5,  # balanced
        }
        return thresholds.get(self.style, 0.5)
    
    def get_adaptation_params(self) -> Dict[str, Any]:
        """
        Get adaptation parameters for agent.
        
        Returns:
            Dictionary of parameters
        """
        params = {
            "takeover": {
                "autonomy_level": "low",
                "explanation_frequency": "high",
                "confirmation_required": True,
            },
            "hands_on": {
                "autonomy_level": "medium",
                "explanation_frequency": "medium",
                "confirmation_required": True,
            },
            "hands_off": {
                "autonomy_level": "high",
                "explanation_frequency": "low",
                "confirmation_required": False,
            },
            "collaborative": {
                "autonomy_level": "medium",
                "explanation_frequency": "medium",
                "confirmation_required": True,
            },
        }
        return params.get(self.style, params["collaborative"])
