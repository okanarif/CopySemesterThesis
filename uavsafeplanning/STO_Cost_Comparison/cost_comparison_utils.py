"""
Utility Functions for Trajectory Optimization

This module contains utility functions for trajectory planning, including:
- Time parametrization utilities (tau ↔ time conversions)
- Future: Corridor utilities, validation helpers, etc.

Author: Trajectory Optimization Framework
Date: 2026
"""

import torch
import numpy as np


# ===========================================
# TIME PARAMETRIZATION UTILITIES
# ===========================================

def tau_to_time(tau: torch.Tensor) -> torch.Tensor:
    """
    Map unconstrained time parameters tau ∈ R to positive durations T ∈ R+.
    Implements T = exp(tau) element-wise.

    Args:
        tau: (...,) shaped torch tensor (e.g. [1, N, 1])

    Returns:
        T: same shape tensor with strictly positive entries
    """
    return torch.exp(tau)


def time_to_tau(T: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Inverse of tau_to_time: tau = log(T).

    Args:
        T: positive durations tensor (e.g. [1, N, 1])
        eps: small clamp to avoid log(0)

    Returns:
        tau: same shape tensor
    """
    T_clamped = T.clamp_min(eps)
    return torch.log(T_clamped)





# ===========================================
# FUTURE UTILITIES (Placeholder)
# ===========================================
# 
# Add corridor utilities, validation helpers, etc. here as needed:
# - point_in_polytope()
# - assign_waypoints_to_corridors()
# - check_corridor_coverage()
# - trajectory_validation()
# - etc.
