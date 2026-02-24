"""
Loss functions for 2D elastic tensor regression with symmetry awareness.

L_total = L_data + λ_sym · L_sym

where:
    L_data = ||Ĉ - C_true||²
    L_sym = ||Ĉ - Π(Ĉ)||²  (symmetry regularization)
"""

import torch
import torch.nn.functional as F
from typing import Optional, List

from .elastic_projection_2d import mixture_projection


def frobenius_norm_loss(pred6: torch.Tensor, target6: torch.Tensor) -> torch.Tensor:
    """
    Frobenius norm (L2) loss for elastic tensors.

    Args:
        pred6: (B, 6)
        target6: (B, 6)

    Returns:
        (B,) loss per sample
    """
    return ((pred6 - target6) ** 2).sum(dim=-1)


def mae_loss(pred6: torch.Tensor, target6: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error (L1) loss.

    Args:
        pred6: (B, 6)
        target6: (B, 6)

    Returns:
        (B,) loss per sample
    """
    return (pred6 - target6).abs().sum(dim=-1)


def symmetry_mixture_loss(
    pred6: torch.Tensor,
    alpha: torch.Tensor,
    shear_convention: str = "gamma"
) -> torch.Tensor:
    """
    Symmetry regularization via mixture projection.

    L_sym = ||Ĉ - Π(Ĉ)||²

    where Π(Ĉ) = Σ_k α_k · Π_Gk(Ĉ)

    Args:
        pred6: (B, 6) predicted tensor
        alpha: (B, 4) symmetry mixture weights
        shear_convention: "gamma" or "epsilon"

    Returns:
        (B,) symmetry loss per sample
    """
    proj6 = mixture_projection(pred6, alpha, shear_convention=shear_convention)
    return frobenius_norm_loss(pred6, proj6)


class ElasticSymmetryLoss(torch.nn.Module):
    """
    Combined loss for symmetry-aware elastic tensor regression.

    L = L_data + λ_sym · L_sym

    Supports:
        - Scheduled λ_sym (increase over training)
        - MSE or MAE for data term
        - Sample weighting (e.g., higher for C2DB labels)
    """

    def __init__(
        self,
        lambda_sym: float = 0.1,
        data_loss_fn: str = "mse",  # "mse" or "mae"
        shear_convention: str = "gamma",
        lambda_sym_schedule: Optional[dict] = None,
    ):
        """
        Args:
            lambda_sym: Initial weight for symmetry term
            data_loss_fn: "mse" (Frobenius) or "mae"
            shear_convention: "gamma" (engineering) or "epsilon" (tensorial)
            lambda_sym_schedule: Optional dict with scheduling params
                Example: {"start": 0.05, "end": 0.3, "warmup_epochs": 50}
        """
        super().__init__()

        self.lambda_sym = lambda_sym
        self.data_loss_fn = data_loss_fn
        self.shear_convention = shear_convention
        self.lambda_sym_schedule = lambda_sym_schedule

        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Update current epoch for scheduling."""
        self.current_epoch = epoch

    def get_lambda_sym(self) -> float:
        """Get current λ_sym value (with scheduling if enabled)."""
        if self.lambda_sym_schedule is None:
            return self.lambda_sym

        start = self.lambda_sym_schedule.get("start", 0.05)
        end = self.lambda_sym_schedule.get("end", 0.3)
        warmup = self.lambda_sym_schedule.get("warmup_epochs", 50)

        # Debug print (remove after verification)
        if self.current_epoch == 0:
            print(f"[DEBUG ElasticSymmetryLoss] lambda_sym_schedule: {self.lambda_sym_schedule}")
            print(f"[DEBUG] start={start}, end={end}, warmup={warmup}")

        if self.current_epoch < warmup:
            # Linear warmup
            progress = self.current_epoch / warmup
            return start + (end - start) * progress
        else:
            return end

    def forward(
        self,
        pred6: torch.Tensor,
        target6: torch.Tensor,
        alpha: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ):
        """
        Compute combined loss.

        Args:
            pred6: (B, 6) predicted tensor
            target6: (B, 6) ground truth tensor
            alpha: (B, 4) symmetry mixture weights
            sample_weights: (B,) optional per-sample weights
                Example: higher for C2DB, lower for Robo-only

        Returns:
            total_loss: scalar
            loss_dict: dict with individual components
        """
        B = pred6.shape[0]

        # Data term
        if self.data_loss_fn == "mse":
            data_loss = frobenius_norm_loss(pred6, target6)
        elif self.data_loss_fn == "mae":
            data_loss = mae_loss(pred6, target6)
        else:
            raise ValueError(f"Unknown data loss: {self.data_loss_fn}")

        # Symmetry term
        lambda_sym_current = self.get_lambda_sym()
        sym_loss = symmetry_mixture_loss(
            pred6, alpha, shear_convention=self.shear_convention
        )

        # Apply sample weights if provided
        if sample_weights is not None:
            data_loss = data_loss * sample_weights
            sym_loss = sym_loss * sample_weights

        # Combine
        total_loss = data_loss.mean() + lambda_sym_current * sym_loss.mean()

        # Metrics
        loss_dict = {
            "loss": total_loss.item(),
            "data_loss": data_loss.mean().item(),
            "sym_loss": sym_loss.mean().item(),
            "lambda_sym": lambda_sym_current,
        }

        return total_loss, loss_dict


class ElasticSymmetryLossWithEntropy(ElasticSymmetryLoss):
    """
    Extended loss with optional entropy regularization on α.

    L = L_data + λ_sym·L_sym + λ_ent·H(α)

    where H(α) = -Σ α_k log α_k

    Use λ_ent > 0 to encourage confident (low entropy) predictions.
    """

    def __init__(
        self,
        lambda_sym: float = 0.1,
        lambda_ent: float = 0.01,
        data_loss_fn: str = "mse",
        shear_convention: str = "gamma",
        lambda_sym_schedule: Optional[dict] = None,
    ):
        super().__init__(
            lambda_sym=lambda_sym,
            data_loss_fn=data_loss_fn,
            shear_convention=shear_convention,
            lambda_sym_schedule=lambda_sym_schedule,
        )
        self.lambda_ent = lambda_ent

    def forward(
        self,
        pred6: torch.Tensor,
        target6: torch.Tensor,
        alpha: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ):
        """
        Compute combined loss with entropy regularization.
        """
        # Get base losses
        total_loss_base, loss_dict = super().forward(
            pred6, target6, alpha, sample_weights
        )

        # Entropy regularization
        if self.lambda_ent > 0:
            eps = 1e-12
            entropy = -(alpha * (alpha + eps).log()).sum(dim=-1)

            if sample_weights is not None:
                entropy = entropy * sample_weights

            ent_loss = entropy.mean()

            # Update total loss
            total_loss = total_loss_base + self.lambda_ent * ent_loss

            # Update metrics
            loss_dict["loss"] = total_loss.item()
            loss_dict["ent_loss"] = ent_loss.item()
            loss_dict["lambda_ent"] = self.lambda_ent

            return total_loss, loss_dict
        else:
            return total_loss_base, loss_dict
