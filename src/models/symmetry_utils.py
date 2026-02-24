"""
Symmetry projection and loss utilities for 2D elastic tensors.

Integrates with existing tensor_trainer.py
"""

import torch
import torch.nn.functional as F


# ============= Projection Functions =============

def proj_triclinic(pred6):
    """No constraints"""
    return pred6


def proj_rectangular(pred6):
    """C16 = C26 = 0"""
    C11, C22, C12, C66, C16, C26 = pred6.unbind(dim=-1)
    C16 = torch.zeros_like(C16)
    C26 = torch.zeros_like(C26)
    return torch.stack([C11, C22, C12, C66, C16, C26], dim=-1)


def proj_tetragonal(pred6):
    """C11 = C22, C16 = C26 = 0"""
    C11, C22, C12, C66, C16, C26 = pred6.unbind(dim=-1)
    C11_avg = 0.5 * (C11 + C22)
    return torch.stack([C11_avg, C11_avg, C12, C66,
                       torch.zeros_like(C16), torch.zeros_like(C26)], dim=-1)


def proj_hexagonal(pred6):
    """C11 = C22, C16 = C26 = 0, C66 = (C11-C12)/2"""
    C11, C22, C12, C66, C16, C26 = pred6.unbind(dim=-1)
    C11_avg = 0.5 * (C11 + C22)
    C66_iso = 0.5 * (C11_avg - C12)
    return torch.stack([C11_avg, C11_avg, C12, C66_iso,
                       torch.zeros_like(C16), torch.zeros_like(C26)], dim=-1)


def mixture_projection(pred6, alpha):
    """
    Π(Ĉ) = Σ α_k · Π_Gk(Ĉ)

    Args:
        pred6: (B, 6)
        alpha: (B, 4) softmax weights
    """
    P = torch.stack([
        proj_triclinic(pred6),
        proj_rectangular(pred6),
        proj_tetragonal(pred6),
        proj_hexagonal(pred6),
    ], dim=1)  # (B, 4, 6)

    return (alpha.unsqueeze(-1) * P).sum(dim=1)


# ============= Robo Priors =============

SYSTEM_TO_PRIOR = {
    "triclinic": [0.60, 0.25, 0.10, 0.05],
    "monoclinic": [0.60, 0.25, 0.10, 0.05],
    "orthorhombic": [0.15, 0.60, 0.20, 0.05],
    "rectangular": [0.15, 0.60, 0.20, 0.05],
    "tetragonal": [0.10, 0.15, 0.60, 0.15],
    "square": [0.10, 0.15, 0.60, 0.15],
    "hexagonal": [0.05, 0.10, 0.20, 0.65],
    "trigonal": [0.05, 0.10, 0.20, 0.65],
    "cubic": [0.05, 0.10, 0.20, 0.65],
}

DEFAULT_PRIOR = [0.25, 0.25, 0.25, 0.25]


def get_alpha_with_prior(alpha_logits, crystal_systems, beta=1.0):
    """
    α = softmax(z + β·log p_prior)

    Args:
        alpha_logits: (B, 4) raw logits
        crystal_systems: list of B system names
        beta: prior strength
    """
    if crystal_systems is None or beta == 0:
        return F.softmax(alpha_logits, dim=-1)

    B = alpha_logits.shape[0]
    device = alpha_logits.device

    # Get priors
    prior_probs = []
    for system in crystal_systems:
        system_lower = system.lower().strip() if isinstance(system, str) else "triclinic"
        prior = SYSTEM_TO_PRIOR.get(system_lower, DEFAULT_PRIOR)
        prior_probs.append(prior)

    prior_probs = torch.tensor(prior_probs, dtype=alpha_logits.dtype, device=device)
    log_prior = torch.log(prior_probs + 1e-12)

    combined_logits = alpha_logits + beta * log_prior
    return F.softmax(combined_logits, dim=-1)


# ============= Loss Functions =============

def symmetry_loss(pred6, alpha):
    """L_sym = ||Ĉ - Π(Ĉ)||²"""
    proj6 = mixture_projection(pred6, alpha)
    return ((pred6 - proj6) ** 2).sum(dim=-1)


def compute_symmetry_aware_loss(
    pred6,
    target6,
    alpha_logits,
    crystal_systems=None,
    beta_prior=1.0,
    lambda_sym=0.1,
    loss_fn="mse"
):
    """
    Complete loss computation.

    L = L_data + λ_sym · L_sym

    Args:
        pred6: (B, 6)
        target6: (B, 6)
        alpha_logits: (B, 4)
        crystal_systems: list of B system names
        beta_prior: Robo prior strength
        lambda_sym: symmetry loss weight
        loss_fn: "mse" or "mae"

    Returns:
        total_loss, metrics_dict
    """
    # Get alpha with prior
    alpha = get_alpha_with_prior(alpha_logits, crystal_systems, beta_prior)

    # Data loss
    if loss_fn == "mse":
        data_loss = ((pred6 - target6) ** 2).sum(dim=-1)
    elif loss_fn == "mae":
        data_loss = (pred6 - target6).abs().sum(dim=-1)
    else:
        data_loss = ((pred6 - target6) ** 2).sum(dim=-1)

    # Symmetry loss
    sym_loss = symmetry_loss(pred6, alpha)

    # Total
    total_loss = data_loss.mean() + lambda_sym * sym_loss.mean()

    metrics = {
        "data_loss": data_loss.mean().item(),
        "sym_loss": sym_loss.mean().item(),
        "alpha_entropy": -(alpha * torch.log(alpha + 1e-12)).sum(dim=-1).mean().item(),
    }

    return total_loss, metrics
