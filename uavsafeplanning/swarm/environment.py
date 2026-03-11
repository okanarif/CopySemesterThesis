from __future__ import annotations

import os
import numpy as np
import pydecomp as pdc

from traj_gen_utils import load_voxel_grid, astar_3d, reduce_turns_by_los


class Environment:
    """Shared environment for multi-UAV planning.

    Wraps the voxel grid, obstacle point cloud, and provides a ``plan_path``
    helper that runs A* -> LOS pruning -> pydecomp corridor generation.
    """

    def __init__(
        self,
        forest_path: str = "traj_gen_utils/data/forest.txt",
        padding: int = 0,
        astar_connectivity: int = 18,
        los_clearance: int = 0,
        box_np: np.ndarray | None = None,
    ):
        self.vg = load_voxel_grid(forest_path, padding=padding)
        self.grid3D = self.vg.grid
        self.origin = np.asarray(self.vg.info.origin, dtype=np.float64)

        occ_idx = np.argwhere(self.grid3D == 1)
        self.obs_np = (occ_idx + 0.5).astype(np.float64) + self.origin

        self.connectivity = astar_connectivity
        self.los_clearance = los_clearance
        self.box_np = box_np if box_np is not None else np.array(
            [[5.0, 5.0, 5.0]], dtype=np.float64
        )

    def plan_path(self, start: np.ndarray, goal: np.ndarray):
        """Run A* -> LOS prune -> pydecomp and return (path_np, A_list, b_list).

        Parameters
        ----------
        start, goal : array-like of shape (3,)
            World-frame coordinates (will be converted to grid indices internally).

        Returns
        -------
        path_np : np.ndarray
            Pruned waypoints in world frame, shape (K, 3).
        A_list, b_list : lists
            Half-space corridor constraints from ``pydecomp``.
        path_indices : list
            Raw A* indices (for visualization).
        """
        start_idx = self.vg.info.to_index(tuple(int(c) for c in start))
        goal_idx = self.vg.info.to_index(tuple(int(c) for c in goal))

        path_indices = astar_3d(
            self.grid3D, start_idx, goal_idx, connectivity=self.connectivity
        )
        if path_indices is None:
            raise RuntimeError(
                f"A* found no path from {start} to {goal}"
            )

        path_pruned = reduce_turns_by_los(
            np.array(path_indices), self.grid3D, clearance=self.los_clearance
        )

        path_np = (np.asarray(path_pruned, np.float64) + 0.5) + self.origin

        A_list, b_list = pdc.convex_decomposition_3D(
            self.obs_np, path_np, self.box_np
        )

        return path_np, A_list, b_list, path_indices
