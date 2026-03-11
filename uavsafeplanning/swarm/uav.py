from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class UAV:
    """Represents a single UAV with start/goal and planning results."""

    uav_id: str
    start: np.ndarray
    goal: np.ndarray
    radius: float = 1.0

    # Populated by the planning pipeline
    path_indices: Optional[List] = field(default=None, repr=False)
    path_np: Optional[np.ndarray] = field(default=None, repr=False)
    A_list: Optional[list] = field(default=None, repr=False)
    b_list: Optional[list] = field(default=None, repr=False)
    sto_results: Optional[dict] = field(default=None, repr=False)

    # Time offset applied by conflict resolution (seconds)
    time_offset: float = 0.0

    def total_time(self) -> float:
        if self.sto_results is None:
            return 0.0
        return float(self.sto_results["total_time"])

    def sample_position(self, t: float) -> Optional[np.ndarray]:
        """Return interpolated position at global time *t* (accounting for offset).

        Returns None if *t* is before this UAV's launch or after its trajectory ends.
        """
        if self.sto_results is None:
            return None
        local_t = t - self.time_offset
        if local_t < 0:
            return self.start.copy()
        total = self.total_time()
        if local_t >= total:
            return self.goal.copy()
        t_eval = self.sto_results["t_eval"]
        pos = self.sto_results["pos"]
        return np.array([
            np.interp(local_t, t_eval, pos[:, d]) for d in range(3)
        ])

    def sample_velocity(self, t: float) -> Optional[np.ndarray]:
        """Return interpolated velocity at global time *t*."""
        if self.sto_results is None:
            return None
        local_t = t - self.time_offset
        if local_t < 0 or local_t >= self.total_time():
            return np.zeros(3)
        t_eval = self.sto_results["t_eval"]
        vel = self.sto_results["vel"]
        return np.array([
            np.interp(local_t, t_eval, vel[:, d]) for d in range(3)
        ])
