"""
Global path planner — A* over the voxel grid for each UAV in the fleet.

Pipeline per UAV
----------------
1. Convert world-space start / goal to voxel indices.
2. Run A* (26-connectivity by default).
3. Prune the raw path with line-of-sight (LOS) reduction — keeps only the
   turn points that cannot be short-cut by a straight free-space segment.
4. Convert the pruned waypoints back to world coordinates (voxel centres).
5. Snap first / last waypoint exactly onto the UAV's start / goal.

Public API
----------
    from global_planner import plan_paths, PlanResult

    results = plan_paths(env, fleet, connectivity=26, clearance=1)
    for r in results:
        print(r.uav_id, r.path_world.shape, r.path_length)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
import yaml

# ── path bootstrap ─────────────────────────────────────────────────────────────
_swarm_dir   = os.path.dirname(os.path.abspath(__file__))
_uavsafe_dir = os.path.dirname(_swarm_dir)
for _p in (_swarm_dir, _uavsafe_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from traj_gen_utils import astar_3d, reduce_turns_by_los
from environment    import Environment
from uav            import Fleet
from logger         import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlannerConfig:
    """Validated planner parameters loaded from planner.yaml."""
    connectivity:     int   = 26
    clearance:        int   = 1
    inflation_radius: float = 0.0
    box_half_extents: list  = field(default_factory=lambda: [5.0, 5.0, 5.0])

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PlannerConfig":
        """Load and validate planner config from a YAML file."""
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}

        # ── A* ────────────────────────────────────────────────────────────────
        astar = raw.get("astar", {})

        connectivity = int(astar.get("connectivity", 26))
        if connectivity not in (6, 18, 26):
            raise ValueError(
                f"planner.yaml: astar.connectivity must be 6, 18, or 26 "
                f"(got {connectivity})"
            )

        clearance = int(astar.get("clearance", 1))
        if clearance < 0:
            raise ValueError(
                f"planner.yaml: astar.clearance must be >= 0 (got {clearance})"
            )

        inflation_radius = float(astar.get("inflation_radius", 0.0))
        if inflation_radius < 0:
            raise ValueError(
                f"planner.yaml: astar.inflation_radius must be >= 0 "
                f"(got {inflation_radius})"
            )

        # ── Safe corridor ─────────────────────────────────────────────────────
        sc  = raw.get("safe_corridor", {})
        bhe = sc.get("box_half_extents", [5.0, 5.0, 5.0])
        if not (isinstance(bhe, list) and len(bhe) == 3):
            raise ValueError(
                "planner.yaml: safe_corridor.box_half_extents must be a list "
                f"of 3 floats, e.g. [5.0, 5.0, 5.0]  (got {bhe})"
            )
        bhe = [float(v) for v in bhe]

        return cls(
            connectivity     = connectivity,
            clearance        = clearance,
            inflation_radius = inflation_radius,
            box_half_extents = bhe,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlanResult:
    """A* planning result for one UAV."""
    uav_id:        str
    path_world:    np.ndarray   # (N, 3) — pruned waypoints in world coords [m]
    n_astar_nodes: int          # raw A* path length before LOS pruning
    n_waypoints:   int          # waypoints after LOS pruning  (= len(path_world))
    path_length:   float        # Euclidean arc length [m]


# ─────────────────────────────────────────────────────────────────────────────
# Main planner
# ─────────────────────────────────────────────────────────────────────────────

def plan_paths(
    env:          Environment,
    fleet:        Fleet,
    connectivity: int = 26,
    clearance:    int = 1,
) -> list[PlanResult]:
    """
    Run A* for every UAV in the fleet and return pruned world-space paths.

    Parameters
    ----------
    env          : Environment — provides the (cached) voxel grid.
    fleet        : Fleet       — provides start / goal for each UAV.
    connectivity : int         — 6, 18, or 26.
                                 26 allows diagonal moves and usually gives
                                 smoother, shorter paths.
    clearance    : int         — voxel buffer kept around obstacles during
                                 LOS pruning.  0 = no buffer, 1 = half a
                                 resolution-length margin (recommended).

    Returns
    -------
    list[PlanResult]
        One entry per UAV, in the same order as ``fleet.uavs``.

    Raises
    ------
    RuntimeError
        If A* finds no path for any UAV.
    """
    vg   = env.to_voxel_grid()   # cached after Cell 1
    w    = env.world
    grid = vg.grid

    results: list[PlanResult] = []

    for uav in fleet.uavs:
        t0 = time.perf_counter()

        # ── world → grid index ────────────────────────────────────────────────
        start_idx = w.world_to_index(uav.start)
        goal_idx  = w.world_to_index(uav.goal)

        # ── A* ────────────────────────────────────────────────────────────────
        raw_path = astar_3d(grid, start_idx, goal_idx, connectivity=connectivity)

        if raw_path is None:
            raise RuntimeError(
                f"A* found no path for '{uav.id}'  "
                f"(start_idx={start_idx}, goal_idx={goal_idx}).  "
                f"Verify that start/goal are inside world bounds and not "
                f"blocked by an obstacle."
            )

        n_raw = len(raw_path)

        # ── LOS pruning ───────────────────────────────────────────────────────
        # Keeps only the waypoints that are true 'turn points' —
        # consecutive collinear voxels are merged into a single segment.
        pruned_idx = reduce_turns_by_los(
            np.array(raw_path, dtype=int), grid, clearance=clearance
        )

        # ── grid index → world coords (voxel centres) ─────────────────────────
        path_world = np.array([
            w.index_to_world(tuple(int(v) for v in idx))
            for idx in pruned_idx
        ])

        # Snap endpoints exactly to start / goal
        # (voxel centres are offset by 0.5 * resolution from the ideal position)
        path_world[0]  = uav.start.copy()
        path_world[-1] = uav.goal.copy()

        # ── path length ───────────────────────────────────────────────────────
        length = float(np.linalg.norm(np.diff(path_world, axis=0), axis=1).sum())

        results.append(PlanResult(
            uav_id        = uav.id,
            path_world    = path_world,
            n_astar_nodes = n_raw,
            n_waypoints   = len(pruned_idx),
            path_length   = length,
        ))

        dt = (time.perf_counter() - t0) * 1000
        log.debug(
            f"{uav.id}:  A*={n_raw} nodes  →  LOS={len(pruned_idx)} waypoints  "
            f"length={length:.1f} m  ({dt:.0f} ms)"
        )

    return results
