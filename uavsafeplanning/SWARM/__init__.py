"""
SWARM — Centralized multi-UAV trajectory planning package.

Public surface for import-time convenience:

    from SWARM.environment import Environment, ConfigValidationError
    from SWARM.logger      import get_logger
"""

from .logger      import get_logger
from .environment import (
    Environment,
    ConfigValidationError,
    WorldBounds,
    CylinderObstacle,
    WallObstacle,
)
from .uav import UAVConfig, Fleet

__all__ = [
    "get_logger",
    "Environment",
    "ConfigValidationError",
    "WorldBounds",
    "CylinderObstacle",
    "WallObstacle",
    "UAVConfig",
    "Fleet",
]
