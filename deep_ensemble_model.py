"""
Deep Ensemble of 3 1D-CNNs for Uncertainty Quantification on Husky A200 Terrain Data.
Implements:
1. SoftmaxClassifier1D: Standard 1D-CNN with identical backbone to EvidentialClassifier1D
2. DeepEnsembleCNN: Ensemble of M=3 independently initialized and trained 1D-CNNs
   with Temperature Scaling and Shannon Entropy / Mutual Information uncertainty decomposition.
"""

from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxClassifier1D(nn.Module):
    """
    Single 1D-CNN member of the Deep Ensemble.
    Uses the exact same convolutional backbone and hidden dimensions as EvidentialClassifier1D
    to ensure a fair architectural comparison.
    """

    def __init__(self, in_channels: int = 10, num_classes: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits of shape (batch_size, num_classes)."""
        feat = self.features(x).squeeze(-1)
        logits = self.classifier(feat)
        return logits


class DeepEnsembleCNN(nn.Module):
    """
    Deep Ensemble of M=3 1D-CNNs (Lakshminarayanan et al., 2017) with Temperature Scaling.
    Provides gold-standard Bayesian-approximate uncertainty quantification:
      - Predictive probabilities: mean softmax across ensemble members
      - Total Uncertainty: Normalized Predictive Shannon Entropy H(p_bar) / ln(K) in [0, 1]
      - Aleatoric Uncertainty: Average member entropy E[H(p_m)] / ln(K) in [0, 1]
      - Epistemic Uncertainty: Mutual Information (BALD) = Total - Aleatoric
      - Member Disagreement: Standard deviation of predicted probabilities across the M=3 models
    """

    def __init__(
        self,
        num_members: int = 3,
        in_channels: int = 10,
        num_classes: int = 5,
        hidden_dim: int = 64,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_members = num_members
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.register_buffer("temperature", torch.tensor(float(temperature)))

        self.members = nn.ModuleList(
            [SoftmaxClassifier1D(in_channels, num_classes, hidden_dim) for _ in range(num_members)]
        )

    def set_temperature(self, temp: float):
        self.temperature.fill_(float(temp))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all M=3 ensemble members.
        Returns mean probabilities of shape (batch_size, num_classes).
        """
        member_probs = []
        for member in self.members:
            logits = member(x) / self.temperature
            probs = F.softmax(logits, dim=-1)
            member_probs.append(probs)

        # Stack: (M, B, K)
        stacked_probs = torch.stack(member_probs, dim=0)
        mean_probs = torch.mean(stacked_probs, dim=0)
        return mean_probs

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Executes all M=3 CNNs and computes comprehensive ensemble uncertainty metrics.
        """
        member_probs = []
        for member in self.members:
            logits = member(x) / self.temperature
            probs = F.softmax(logits, dim=-1)
            member_probs.append(probs)

        # Shape: (M, B, K)
        stacked_probs = torch.stack(member_probs, dim=0)

        # 1. Mean predictive distribution: p_bar = (1/M) sum p_m
        mean_probs = torch.mean(stacked_probs, dim=0)  # (B, K)
        confidence, preds = torch.max(mean_probs, dim=1)

        # 2. Normalized Total Predictive Entropy: H(p_bar) / ln(K) in [0, 1]
        eps = 1e-8
        max_entropy = math.log(self.num_classes)
        total_entropy = -torch.sum(mean_probs * torch.log(mean_probs + eps), dim=1)
        norm_total_uncertainty = total_entropy / max_entropy

        # 3. Normalized Aleatoric Uncertainty: (1/M) sum H(p_m) / ln(K)
        member_entropies = -torch.sum(stacked_probs * torch.log(stacked_probs + eps), dim=2)  # (M, B)
        aleatoric_uncertainty = torch.mean(member_entropies, dim=0) / max_entropy

        # 4. Epistemic Uncertainty (Mutual Information / BALD): H(p_bar) - E[H(p_m)]
        epistemic_mi = torch.clamp(norm_total_uncertainty - aleatoric_uncertainty, min=0.0)

        # 5. Ensemble Disagreement (Variance / Std across members)
        prob_std = torch.std(stacked_probs, dim=0).mean(dim=1)  # (B,)

        return {
            "probs": mean_probs,
            "member_probs": stacked_probs,
            "preds": preds,
            "confidence": confidence,
            "uncertainty": norm_total_uncertainty,      # Primary normalized uncertainty in [0, 1]
            "epistemic_mi": epistemic_mi,               # Mutual information in [0, 1]
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "disagreement_std": prob_std,
        }
