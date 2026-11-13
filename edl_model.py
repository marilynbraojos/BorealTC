import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class EvidentialClassifier1D(nn.Module):
    """
    1D Convolutional Neural Network for time-series classification with 
    Evidential Deep Learning (EDL) for uncertainty quantification.
    
    Instead of standard softmax probabilities, this network outputs non-negative
    evidence values that parameterize a Dirichlet prior distribution over classes.
    """

    def __init__(self, in_channels: int = 10, num_classes: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes

        # Feature extractor for multi-sensor time-series (IMU + proprioception)
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

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes forward pass and returns non-negative evidence for each class.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_len)
            
        Returns:
            torch.Tensor: Evidence tensor e >= 0 of shape (batch_size, num_classes)
        """
        feat = self.features(x)
        feat = feat.squeeze(-1)
        logits = self.classifier(feat)
        
        # Softplus ensures evidence is strictly non-negative with smooth gradients
        evidence = F.softplus(logits)
        return evidence

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Runs inference and computes classification probabilities and uncertainty metrics.
        
        Returns:
            Dict containing:
                - 'evidence': non-negative evidence vector e_k
                - 'alpha': Dirichlet parameters alpha_k = e_k + 1
                - 'probs': Expected class probabilities p_k = alpha_k / S
                - 'uncertainty': Epistemic uncertainty / vacuity u = K / S (in [0, 1])
                - 'confidence': Max probability max_k(p_k)
                - 'preds': Predicted class indices argmax(p_k)
        """
        evidence = self.forward(x)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / S
        uncertainty = self.num_classes / S.squeeze(1)
        confidence, preds = torch.max(probs, dim=1)

        return {
            "evidence": evidence,
            "alpha": alpha,
            "probs": probs,
            "uncertainty": uncertainty,
            "confidence": confidence,
            "preds": preds,
        }


def kl_divergence_dirichlet(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Computes KL divergence between predicted Dirichlet and uniform Dirichlet prior Dir(1).
    Used to penalize misleading evidence for non-ground-truth classes.
    """
    ones = torch.ones_like(alpha)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    sum_ones = torch.tensor(num_classes, dtype=alpha.dtype, device=alpha.device)

    term1 = torch.lgamma(sum_alpha) - torch.lgamma(sum_ones)
    term2 = torch.sum(torch.lgamma(ones) - torch.lgamma(alpha), dim=1, keepdim=True)
    term3 = torch.sum(
        (alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(sum_alpha)),
        dim=1,
        keepdim=True,
    )
    return (term1 + term2 + term3).squeeze(1)


def edl_loss(
    evidence: torch.Tensor,
    targets: torch.Tensor,
    epoch: int,
    num_classes: int,
    annealing_epochs: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evidential Deep Learning Loss (Sensoy et al., 2018).
    
    Combines expected Bayesian cross-entropy under the Dirichlet distribution
    with an annealed KL-divergence regularizer to minimize counter-evidence.
    """
    # 1. Convert integer targets to one-hot encoding
    y_one_hot = F.one_hot(targets, num_classes=num_classes).float()

    # 2. Dirichlet parameters
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)

    # 3. Expected Cross-Entropy Loss: sum y_k * (digamma(S) - digamma(alpha_k))
    loss_ace = torch.sum(y_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)

    # 4. KL Divergence Regularization for incorrect classes
    # Modulate alpha so target class evidence is not penalized
    alpha_tilde = y_one_hot + (1.0 - y_one_hot) * alpha
    kl = kl_divergence_dirichlet(alpha_tilde, num_classes)

    # Anneal KL weight from 0 to 1 over annealing_epochs
    annealing_coef = min(1.0, float(epoch) / float(max(1, annealing_epochs)))
    total_loss = torch.mean(loss_ace + annealing_coef * kl)

    return total_loss, torch.mean(loss_ace), torch.mean(kl)
