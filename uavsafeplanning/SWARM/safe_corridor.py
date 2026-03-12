"""
Safe Flight Corridor (SFC) extraction via convex decomposition.

For each UAV the pruned A* waypoints are passed together with the full
obstacle point cloud to `pydecomp.convex_decomposition_3D`, which returns
a list of convex polytopes in half-space form  (A @ x <= b)  that form a
collision-free corridor enclosing the path.

This is the same algorithm used in the STO pipeline; the only difference is
how `obs_np` and `path_np` are built to account for our metric resolution.

Public API
----------
    from safe_corridor import extract_corridors, CorridorResult

    corridors = extract_corridors(env, plan_results,
                                  box_half_extents=[5.0, 5.0, 5.0])
    for cr in corridors:
        print(cr.uav_id, len(cr.A_list), "polytopes")
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np

# ── path bootstrap ─────────────────────────────────────────────────────────────
_swarm_dir   = os.path.dirname(os.path.abspath(__file__))
_uavsafe_dir = os.path.dirname(_swarm_dir)
for _p in (_swarm_dir, _uavsafe_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pydecomp as pdc
from environment import Environment
from logger      import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CorridorResult:
    """Safe corridor decomposition for one UAV."""
    uav_id:    str
    A_list:    list          # list of np.ndarray (n_c, 3) — normal vectors
    b_list:    list          # list of np.ndarray (n_c,)   — offsets  (A@x <= b)
    waypoints: np.ndarray   # path waypoints that define the segments (N, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_obs_np(env: Environment) -> np.ndarray:
    """
    Obstacle point cloud: occupied voxel centres in world coordinates [m].

    Voxel centre formula:
        world_pos = (grid_index + 0.5) * resolution + world_origin
    """
    vg     = env.to_voxel_grid()
    w      = env.world
    occ    = np.argwhere(vg.grid).astype(np.float64)       # (M, 3) grid indices
    origin = np.array([w.x_min, w.y_min, w.z_min], dtype=np.float64)
    return (occ + 0.5) * float(w.resolution) + origin


# ─────────────────────────────────────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_corridors(
    env:              Environment,
    plan_results:     list,                          # list[PlanResult]
    box_half_extents: list | np.ndarray = (5.0, 5.0, 5.0),
) -> list[CorridorResult]:
    """
    Extract a safe flight corridor for every UAV.

    Parameters
    ----------
    env              : Environment  — used to build the obstacle point cloud.
    plan_results     : list[PlanResult]  — output from global_planner.plan_paths().
    box_half_extents : (bx, by, bz) — half-sizes of the seed box used as the
                       initial corridor estimate around each path segment [m].
                       The algorithm inflates / shrinks the box to fit free
                       space.  Corresponds directly to ``box_np`` in the STO
                       pipeline.

    Returns
    -------
    list[CorridorResult]
        One entry per UAV, in the same order as ``plan_results``.

    Raises
    ------
    RuntimeError  — if pydecomp fails for any UAV.
    """
    obs_np = _build_obs_np(env)
    box_np = np.asarray(box_half_extents, dtype=np.float64).reshape(1, 3)

    results: list[CorridorResult] = []

    for pr in plan_results:
        t0      = time.perf_counter()
        path_np = pr.path_world.astype(np.float64)   # (N, 3) — already in metres

        try:
            A_list, b_list = pdc.convex_decomposition_3D(obs_np, path_np, box_np)
        except Exception as exc:
            raise RuntimeError(
                f"Convex decomposition failed for '{pr.uav_id}': {exc}"
            ) from exc

        results.append(CorridorResult(
            uav_id    = pr.uav_id,
            A_list    = A_list,
            b_list    = b_list,
            waypoints = path_np,
        ))

        dt = (time.perf_counter() - t0) * 1000
        avg_c = (sum(np.asarray(A).shape[0] for A in A_list) / len(A_list)
                 if A_list else 0)
        log.debug(
            f"{pr.uav_id}:  {len(A_list)} corridors  "
            f"avg {avg_c:.0f} constraints  ({dt:.0f} ms)"
        )

    return results
