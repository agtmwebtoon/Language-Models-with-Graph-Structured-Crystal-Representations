"""
Symmetry projection for 2D elastic tensors (Voigt notation).

6 DOF parameterization: [C11, C22, C12, C66, C16, C26]

Full 2D elastic tensor (Voigt 3x3):
    [[C11, C12, C16],
     [C12, C22, C26],
     [C16, C26, C66]]

Crystal systems (K=4):
    0. Triclinic: no constraints
    1. Rectangular (Orthorhombic): C16 = C26 = 0
    2. Tetragonal (Square): C11 = C22, C16 = C26 = 0
    3. Hexagonal (Isotropic 2D): above + C66 = (C11 - C12)/2
"""

import torch
import torch.nn.functional as F


def proj_triclinic(pred6: torch.Tensor) -> torch.Tensor:
    """
    Triclinic: No constraints.
    All 6 components are independent.
    """
    return pred6


def proj_rectangular(pred6: torch.Tensor) -> torch.Tensor:
    """
    Rectangular/Orthorhombic: C16 = C26 = 0
    C11, C22, C12, C66 are free.
    """
    C11, C22, C12, C66, C16, C26 = pred6.unbind(dim=-1)
    C16 = torch.zeros_like(C16)
    C26 = torch.zeros_like(C26)
    return torch.stack([C11, C22, C12, C66, C16, C26], dim=-1)


def proj_tetragonal(pred6: torch.Tensor) -> torch.Tensor:
    """
    Tetragonal/Square: C11 = C22, C16 = C26 = 0
    C12, C66 are free.
    """
    C11, C22, C12, C66, C16, C26 = pred6.unbind(dim=-1)

    # Enforce C11 = C22
    C11_avg = 0.5 * (C11 + C22)
    C11 = C11_avg
    C22 = C11_avg

    # Set shear-coupling terms to zero
    C16 = torch.zeros_like(C16)
    C26 = torch.zeros_like(C26)

    return torch.stack([C11, C22, C12, C66, C16, C26], dim=-1)


def proj_hexagonal(pred6: torch.Tensor, shear_convention: str = "epsilon") -> torch.Tensor:
    """
    Hexagonal/Isotropic 2D:
        C11 = C22 (in-plane isotropy)
        C16 = C26 = 0
        C66 = (C11 - C12) / 2  (engineering shear strain convention)
             or = (C11 - C12)  (tensorial strain convention)

    Args:
        shear_convention: "gamma" (engineering) or "epsilon" (tensorial)
    """
    C11, C22, C12, C66, C16, C26 = pred6.unbind(dim=-1)

    # Enforce C11 = C22
    C11_avg = 0.5 * (C11 + C22)
    C11 = C11_avg
    C22 = C11_avg

    # Set shear-coupling to zero
    C16 = torch.zeros_like(C16)
    C26 = torch.zeros_like(C26)

    # Enforce isotropic shear modulus relation
    if shear_convention == "gamma":
        C66 = 0.5 * (C11 - C12)
    elif shear_convention == "epsilon":
        C66 = (C11 - C12)
    else:
        raise ValueError(f"Unknown shear_convention: {shear_convention}")

    return torch.stack([C11, C22, C12, C66, C16, C26], dim=-1)


# Mapping from crystal system names to projection functions
CRYSTAL_SYSTEM_PROJECTIONS = {
    "triclinic": proj_triclinic,
    "monoclinic": proj_triclinic,  # Conservative: treat as triclinic
    "orthorhombic": proj_rectangular,
    "rectangular": proj_rectangular,
    "tetragonal": proj_tetragonal,
    "square": proj_tetragonal,
    "hexagonal": proj_hexagonal,
    "trigonal": proj_hexagonal,  # Similar to hexagonal for 2D
}


# Robo-based symmetry priors (soft physical hints)
SYSTEM_TO_PRIOR = {
    "triclinic": [0.60, 0.25, 0.10, 0.05],     # [tri, rect, tetra, hexa]
    "monoclinic": [0.60, 0.25, 0.10, 0.05],    # Similar to triclinic
    "orthorhombic": [0.15, 0.60, 0.20, 0.05],  # Prefer rectangular
    "rectangular": [0.15, 0.60, 0.20, 0.05],
    "tetragonal": [0.10, 0.15, 0.60, 0.15],    # Prefer tetragonal
    "square": [0.10, 0.15, 0.60, 0.15],
    "hexagonal": [0.05, 0.10, 0.20, 0.65],     # Prefer hexagonal
    "trigonal": [0.05, 0.10, 0.20, 0.65],
    "cubic": [0.05, 0.10, 0.20, 0.65],         # Map to hexagonal (isotropic)
}

# Default uniform prior if system unknown
DEFAULT_PRIOR = [0.25, 0.25, 0.25, 0.25]


def get_symmetry_prior(crystal_system: str) -> list:
    """
    Get prior distribution over 4 symmetry candidates.

    Args:
        crystal_system: Robo-parsed crystal system name

    Returns:
        [p_tri, p_rect, p_tetra, p_hexa] prior probabilities
    """
    system_lower = crystal_system.lower().strip()
    return SYSTEM_TO_PRIOR.get(system_lower, DEFAULT_PRIOR)


def mixture_projection(
    pred6: torch.Tensor,
    alpha: torch.Tensor,
    shear_convention: str = "gamma"
) -> torch.Tensor:
    """
    Apply mixture of symmetry projections weighted by alpha.

    Π(Ĉ) = Σ_k α_k · Π_Gk(Ĉ)

    Args:
        pred6: (B, 6) = [C11, C22, C12, C66, C16, C26]
        alpha: (B, 4) softmax weights [tri, rect, tetra, hexa]
        shear_convention: "gamma" or "epsilon"

    Returns:
        (B, 6) projected tensor = effective elastic tensor
    """
    # Apply each projection
    p0 = proj_triclinic(pred6)
    p1 = proj_rectangular(pred6)
    p2 = proj_tetragonal(pred6)
    p3 = proj_hexagonal(pred6, shear_convention=shear_convention)

    # Stack: (B, 4, 6)
    P = torch.stack([p0, p1, p2, p3], dim=1)

    # Weighted sum: (B, 4, 1) * (B, 4, 6) -> (B, 6)
    alpha_expanded = alpha.unsqueeze(-1)
    return (alpha_expanded * P).sum(dim=1)


def apply_single_projection(
    pred6: torch.Tensor,
    crystal_system: str,
    shear_convention: str = "gamma"
) -> torch.Tensor:
    """
    Apply a single symmetry projection.

    Args:
        pred6: (B, 6)
        crystal_system: name of crystal system
        shear_convention: for hexagonal projection

    Returns:
        (B, 6) projected tensor
    """
    system_lower = crystal_system.lower().strip()
    proj_fn = CRYSTAL_SYSTEM_PROJECTIONS.get(system_lower, proj_triclinic)

    if system_lower in ["hexagonal", "trigonal"]:
        return proj_fn(pred6, shear_convention=shear_convention)
    else:
        return proj_fn(pred6)


def reconstruct_voigt_3x3(pred6: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct 3x3 Voigt elastic tensor from 6 components.

    Args:
        pred6: (B, 6) = [C11, C22, C12, C66, C16, C26]

    Returns:
        (B, 3, 3) Voigt tensor:
            [[C11, C12, C16],
             [C12, C22, C26],
             [C16, C26, C66]]
    """
    B = pred6.shape[0]
    tensor = torch.zeros(B, 3, 3, device=pred6.device, dtype=pred6.dtype)

    C11, C22, C12, C66, C16, C26 = pred6.unbind(dim=-1)

    # Fill symmetric Voigt tensor
    tensor[:, 0, 0] = C11
    tensor[:, 1, 1] = C22
    tensor[:, 2, 2] = C66
    tensor[:, 0, 1] = C12
    tensor[:, 1, 0] = C12
    tensor[:, 0, 2] = C16
    tensor[:, 2, 0] = C16
    tensor[:, 1, 2] = C26
    tensor[:, 2, 1] = C26

    return tensor
