"""Space-time conflict detection and resolution for multi-UAV trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .uav import UAV


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Conflict:
    """A detected conflict between two UAVs over a time window."""

    uav_i_id: str
    uav_j_id: str
    t_start: float
    t_end: float
    min_distance: float


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_conflicts(
    uavs: List[UAV],
    dt: float = 0.1,
    safety_margin: float = 0.0,
) -> List[Conflict]:
    """Check all UAV pairs for space-time proximity violations.

    Two UAVs *i*, *j* conflict at time *t* when
    ``||pos_i(t) - pos_j(t)|| < r_i + r_j + safety_margin``.

    Consecutive conflicting time steps are merged into a single
    ``Conflict`` window ``[t_start, t_end]``.

    Parameters
    ----------
    uavs : list of UAV
        Each UAV must have ``sto_results`` populated (i.e. planned).
    dt : float
        Sampling resolution (seconds).
    safety_margin : float
        Extra clearance beyond the sum of radii.

    Returns
    -------
    list of Conflict
    """
    if len(uavs) < 2:
        return []

    t_max = max(u.time_offset + u.total_time() for u in uavs)
    times = np.arange(0.0, t_max + dt, dt)

    # Pre-sample all positions
    positions = {}
    for u in uavs:
        positions[u.uav_id] = np.array([u.sample_position(t) for t in times])

    conflicts: List[Conflict] = []

    for i in range(len(uavs)):
        for j in range(i + 1, len(uavs)):
            ui, uj = uavs[i], uavs[j]
            threshold = ui.radius + uj.radius + safety_margin

            pos_i = positions[ui.uav_id]
            pos_j = positions[uj.uav_id]
            dists = np.linalg.norm(pos_i - pos_j, axis=1)

            in_conflict = dists < threshold

            # Merge consecutive conflict steps into windows
            _append_conflict_windows(
                conflicts, in_conflict, times, dists,
                ui.uav_id, uj.uav_id, threshold,
            )

    return conflicts


def _append_conflict_windows(
    conflicts: List[Conflict],
    mask: np.ndarray,
    times: np.ndarray,
    dists: np.ndarray,
    id_i: str,
    id_j: str,
    threshold: float,  # noqa: ARG001 – kept for potential future use
) -> None:
    """Merge consecutive True entries in *mask* into Conflict windows."""
    in_window = False
    t_start = 0.0
    min_d = float("inf")

    for k in range(len(mask)):
        if mask[k]:
            if not in_window:
                in_window = True
                t_start = times[k]
                min_d = dists[k]
            else:
                min_d = min(min_d, dists[k])
        else:
            if in_window:
                conflicts.append(Conflict(id_i, id_j, t_start, times[k - 1], min_d))
                in_window = False
    if in_window:
        conflicts.append(Conflict(id_i, id_j, t_start, times[len(mask) - 1], min_d))


def compute_min_distances(
    uavs: List[UAV],
    dt: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-pair minimum distance over time.

    Returns
    -------
    times : np.ndarray, shape (T,)
    pair_dists : np.ndarray, shape (n_pairs, T)
        Row order follows the upper-triangle iteration
        ``(0,1), (0,2), ..., (n-2, n-1)``.
    """
    t_max = max(u.time_offset + u.total_time() for u in uavs)
    times = np.arange(0.0, t_max + dt, dt)

    positions = {
        u.uav_id: np.array([u.sample_position(t) for t in times]) for u in uavs
    }

    pair_dists = []
    for i in range(len(uavs)):
        for j in range(i + 1, len(uavs)):
            d = np.linalg.norm(
                positions[uavs[i].uav_id] - positions[uavs[j].uav_id], axis=1
            )
            pair_dists.append(d)

    return times, np.array(pair_dists) if pair_dists else np.empty((0, len(times)))


# ---------------------------------------------------------------------------
# Resolution -- Strategy A: priority-based time shifting
# ---------------------------------------------------------------------------

def resolve_conflicts_time_shift(
    uavs: List[UAV],
    dt: float = 0.1,
    safety_margin: float = 0.0,
    max_rounds: int = 20,
    verbose: bool = False,
) -> List[Conflict]:
    """Iteratively resolve conflicts by delaying lower-priority UAVs.

    Priority is determined by list order: ``uavs[0]`` has highest priority
    and its trajectory is never shifted.  When a conflict between UAV *i*
    (higher priority) and UAV *j* (lower priority) is found, UAV *j* is
    delayed so it departs after UAV *i* has cleared the conflict window,
    plus a buffer equal to ``(r_i + r_j + safety_margin) / v_avg``.

    Parameters
    ----------
    uavs : list of UAV
        Must be pre-planned (``sto_results`` populated).
    dt, safety_margin : float
        Forwarded to :func:`detect_conflicts`.
    max_rounds : int
        Maximum resolution iterations.
    verbose : bool
        Print progress.

    Returns
    -------
    remaining : list of Conflict
        Conflicts still present after resolution (ideally empty).
    """
    for round_idx in range(max_rounds):
        conflicts = detect_conflicts(uavs, dt=dt, safety_margin=safety_margin)
        if not conflicts:
            if verbose:
                print(f"  [resolve] Round {round_idx}: no conflicts -- done.")
            return []

        if verbose:
            print(f"  [resolve] Round {round_idx}: {len(conflicts)} conflict(s)")

        id_to_uav = {u.uav_id: u for u in uavs}
        id_to_priority = {u.uav_id: idx for idx, u in enumerate(uavs)}

        for c in conflicts:
            pri_i = id_to_priority[c.uav_i_id]
            pri_j = id_to_priority[c.uav_j_id]

            if pri_i < pri_j:
                high, low = id_to_uav[c.uav_i_id], id_to_uav[c.uav_j_id]
            else:
                high, low = id_to_uav[c.uav_j_id], id_to_uav[c.uav_i_id]

            clearance_time = (high.radius + low.radius + safety_margin) / max(
                _avg_speed(high), 0.5
            )
            new_offset = c.t_end + clearance_time
            if new_offset > low.time_offset:
                low.time_offset = new_offset
                if verbose:
                    print(
                        f"    -> delay {low.uav_id} to t_offset={low.time_offset:.2f}s"
                    )

    return detect_conflicts(uavs, dt=dt, safety_margin=safety_margin)


def _avg_speed(uav: UAV) -> float:
    if uav.sto_results is None:
        return 1.0
    vel_norm = uav.sto_results.get("vel_norm")
    if vel_norm is None or len(vel_norm) == 0:
        return 1.0
    return float(np.mean(vel_norm))
