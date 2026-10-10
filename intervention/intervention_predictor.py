"""
Intervention Predictor: LSTM-based model for predicting human intervention timing.

Problem: Autonomous agents either ping too often (annoying) or too rarely (unreliable).
Solution: LSTM-based predictor that learns from action sequences, context, and user history.

Metrics:
- AUC-ROC: Area under ROC curve (discrimination ability)
- Brier Score: Calibration error (lower is better)
- Perfect Timing Score (PTS): Fraction of correct intervention predictions

Research Backing:
- Alibaba Field Experiment (arXiv:2605.14830): Human intervention effectiveness
- Oversight Game (arXiv:2510.26752): Autonomy-safety balance
- Human Oversight Framework (arXiv:2606.05391): Early intervention for meaningful control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import os
from sklearn.linear_model import LogisticRegression
import time


@dataclass
class ActionFeature:
    """
    Feature vector for a single action.
    """
    action_type: str
    action_data: Dict[str, Any]
    context: Dict[str, Any]
    user_history: Dict[str, Any]


class ActionEncoder(nn.Module):
    def __init__(self, action_type_dim: int = 7, data_dim: int = 32, context_dim: int = 16):
        super().__init__()
        self.action_type_embed = nn.Embedding(action_type_dim, 16)
        self.data_encoder = nn.Linear(data_dim, 32)
        self.context_encoder = nn.Linear(context_dim, 16)
        self.output_dim = 64

    def forward(self, action: ActionFeature) -> torch.Tensor:
        action_type_idx = self._action_type_to_idx(action.action_type)
        type_embed = self.action_type_embed(torch.tensor([action_type_idx]))

        data_vector = self._extract_data_features(action.action_data)
        data_embed = F.relu(self.data_encoder(data_vector.unsqueeze(0)))

        context_vector = self._extract_context_features(action.context)
        context_embed = F.relu(self.context_encoder(context_vector.unsqueeze(0)))

        combined = torch.cat([type_embed, data_embed, context_embed], dim=-1)
        return combined.squeeze(0)

    def _action_type_to_idx(self, action_type: str) -> int:
        action_types = ["click", "type", "navigate", "submit", "wait", "fill_form", "confirm"]
        return action_types.index(action_type) if action_type in action_types else 6

    def _extract_data_features(self, data: Dict[str, Any]) -> torch.Tensor:
        amount = data.get("amount", 0)
        normalized_amount = min(float(amount) / 2000.0, 1.0) if isinstance(amount, (int, float)) else 0.0
        features = [
            len(str(data)),
            sum(1 for v in data.values() if isinstance(v, (int, float))),
            normalized_amount,
        ] + [0.0] * (32 - 3)
        return torch.tensor(features[:32], dtype=torch.float32)

    def _extract_context_features(self, context: Dict[str, Any]) -> torch.Tensor:
        task_types = ["bill_payment", "appointment", "paperwork", "other"]
        features = [
            task_types.index(context.get("task_type", "other")) if context.get("task_type") in task_types else 3,
            context.get("complexity", 0.5),
            context.get("risk_level", 0.5),
        ] + [0.0] * (16 - 3)
        return torch.tensor(features[:16], dtype=torch.float32)


class InterventionPredictorModel(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.action_encoder = ActionEncoder()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, action_sequence: List[ActionFeature]) -> torch.Tensor:
        embeddings = [self.action_encoder(action) for action in action_sequence]
        embeddings = torch.stack(embeddings)
        embeddings = embeddings.unsqueeze(0)

        lstm_out, (h_n, c_n) = self.lstm(embeddings)

        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)

        out = self.dropout(F.relu(self.fc1(context)))
        out = torch.sigmoid(self.fc2(out))
        return out.squeeze()


class InterventionPredictor:
    def __init__(self, model_path: str = "./intervention/model.pt", calibrate: bool = False, calibration_data=None):
        self.model_path = model_path
        self.model = InterventionPredictorModel()
        self.weights_loaded = False
        self._load_weights()
        self.model.eval()

        self.calibrator = None
        if calibrate:
            if calibration_data is None:
                raise ValueError("Calibration requires held-out model scores and real labels")
            self._train_calibrator(*calibration_data)

    def _load_weights(self):
        """
        Load pre-trained weights.
        """
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location="cpu", weights_only=True))
                self.weights_loaded = True
            except Exception as e:
                print("Warning: incompatible weights; neural predictions are unavailable.")
        else:
            print(f"Warning: No weights found at {self.model_path}; neural predictions are unavailable.")

    def _train_calibrator(self, scores, labels):
        X = np.asarray(scores, dtype=float).reshape(-1, 1)
        y = np.asarray(labels)
        if len(X) != len(y) or len(y) < 4 or set(y.tolist()) != {0, 1}:
            raise ValueError("Calibration requires matching scores and both label classes")
        if not np.isfinite(X).all() or (X < 0).any() or (X > 1).any():
            raise ValueError("Calibration scores must be finite values in [0, 1]")
        self.calibrator = LogisticRegression(random_state=42)
        self.calibrator.fit(X, y)

    def predict(self, action_sequence: List[ActionFeature]) -> float:
        if not self.weights_loaded:
            raise RuntimeError("No compatible weights loaded; train or provide a model artifact first")
        if not action_sequence:
            raise ValueError("Action sequence cannot be empty")
        with torch.no_grad():
            prob = self.model(action_sequence).item()

        if self.calibrator:
            prob_calibrated = self.calibrator.predict_proba([[prob]])[0, 1]
            return prob_calibrated
        return prob

    def should_intervene(self, action_sequence: List[ActionFeature], threshold: float = 0.5) -> bool:
        if not self.weights_loaded:
            return True  # Missing model: require human review, never use random weights.
        prob = self.predict(action_sequence)
        return prob > threshold

    def get_metrics(self, predictions: List[float], actuals: List[bool]) -> Dict[str, float]:
        from sklearn.metrics import roc_auc_score, brier_score_loss

        if not predictions or len(predictions) != len(actuals):
            raise ValueError("Metrics require nonempty matching predictions and labels")
        auc = roc_auc_score(actuals, predictions) if len(set(actuals)) > 1 else None
        brier = brier_score_loss(actuals, predictions)
        pts = self._compute_pts(predictions, actuals)

        return {
            "auc_roc": auc,
            "brier_score": brier,
            "pts": pts,  # Legacy name; this measures classification accuracy, not timing.
            "classification_accuracy": pts,
        }

    def _compute_pts(self, predictions: List[float], actuals: List[bool]) -> float:
        if not predictions or len(predictions) != len(actuals):
            raise ValueError("Accuracy requires nonempty matching predictions and labels")
        correct = sum(1 for p, a in zip(predictions, actuals) if (p > 0.5) == a)
        return correct / len(predictions)