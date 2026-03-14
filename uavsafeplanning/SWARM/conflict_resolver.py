"""
Temporal conflict resolver for multi-UAV swarm trajectories.

Algorithm
---------
For each replan round:

1. Sort all ConflictEvents by t_start (earliest violation first).
   For pairs with multiple events, keep only the earliest per pair.

2. Per event — determine winner / loser:
   Integrate the path length each UAV travels during the violation window
   [t_start, t_end] by summing Euclidean distances between consecutive
   position samples.  The drone that covered *more ground* (i.e. was moving
   faster through the conflict zone) is the **winner** — its trajectory is
   left unchanged.  The slower drone is the **loser** and will be re-timed.

   Winner/loser assignments are locked after the first round so that the
   same pair always produces the same result across rounds.

3. Compute the required delay for the loser:
   Δt = max(violation_duration, min_delta_t)
   where violation_duration = t_end - t_start.

4. Find the segment in the loser's trajectory that *contains* t_start
   (using the stored per-segment time_allocation).  Distribute Δt across
   segments 0 … k_pre proportionally to their current durations so that
   no single short segment is disproportionately stretched.

5. Re-run STO for the loser with:
   - temporal_only = True        (waypoints frozen)
   - lambda_sep    = 0           (no separation penalty)
   - time_init_override          (distributed time allocation from step 4)
   - frozen_segment_indices = [0..k_pre]
       → tau[0..k_pre] locked at the increased values — the injected delay
         cannot be optimised away
       → tau[k_pre+1..end] free — post-conflict segments can adjust to
         restore corridor / kinematic compliance
   - lambda_time_replan          (weak time cost on free segments)

6. Update traj_map with the new trajectory.

7. Re-detect conflicts.  Repeat up to max_replan_rounds.

Public API
----------
    from conflict_resolver import ReplanConfig, resolve_conflicts

    rp_cfg = ReplanConfig.from_yaml("configs/planner.yaml")
    new_trajs, final_report = resolve_conflicts(
        fleet, trajectories, corridors, paths,
        conflict_report, cd_cfg, sto_cfg, rp_cfg,
    )
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple, Union

import numpy as np
import yaml

# ── path bootstrap ─────────────────────────────────────────────────────────────
_swarm_dir   = os.path.dirname(os.path.abspath(__file__))
_uavsafe_dir = os.path.dirname(_swarm_dir)
for _p in (_swarm_dir, _uavsafe_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from logger import get_logger
from conflict_detector import (
    ConflictDetectorConfig,
    ConflictEvent,
    ConflictReport,
    detect_conflicts,
)
from trajectory_generator import (
    TrajectoryResult,
    STOConfig,
    replan_single_trajectory,
)

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReplanConfig:
    """
    Hyper-parameters for the temporal conflict-resolution loop,
    loaded from the ``replanning:`` block of ``planner.yaml``.

    Attributes
    ----------
    max_replan_rounds : int
        Maximum detect → replan iterations.
    min_delta_t : float
        Minimum delay added to the loser's pre-violation segment [s].
        Guards against degenerate zero-duration violations.
    delta_t_scale : float
        Multiplier applied to the raw violation duration:
        ``Δt = max(violation_duration * delta_t_scale, min_delta_t)``.
        1.0 is fully conservative (full violation window).
        0.6–0.8 is usually sufficient; pair with more ``max_replan_rounds``
        so that any residual violation is cleaned up in a subsequent round.

    Notes
    -----
    ``lambda_time_replan`` is applied only to the **free** post-conflict
    segments (k_pre+1 … end).  Keep it small (≤ 0.05) so the optimizer
    does not aggressively shorten those segments at the expense of corridor
    compliance.  The pre-conflict segments (0 … k_pre) are frozen and
    therefore unaffected by this weight.
    """
    lambda_time_replan: float = 0.01
    max_replan_rounds:  int   = 4
    replan_max_iter:    int   = 50
    min_delta_t:        float = 0.3
    delta_t_scale:      float = 0.7

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ReplanConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        rp = raw.get("replanning", {})
        return cls(
            lambda_time_replan = float(rp.get("lambda_time_replan", 0.01)),
            max_replan_rounds  = int(rp.get("max_replan_rounds",    4)),
            replan_max_iter    = int(rp.get("replan_max_iter",      50)),
            min_delta_t        = float(rp.get("min_delta_t",        0.3)),
            delta_t_scale      = float(rp.get("delta_t_scale",      0.7)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _path_length_in_window(
    traj: TrajectoryResult,
    t_start: float,
    t_end: float,
) -> float:
    """
    Sum of Euclidean distances between consecutive position samples that
    fall inside the time window [t_start, t_end].

    Using arc-length rather than endpoint displacement correctly handles
    curved or oscillating paths.
    """
    mask = (traj.t_eval >= t_start) & (traj.t_eval <= t_end)
    pos_seg = traj.pos[mask]
    if len(pos_seg) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pos_seg, axis=0), axis=1)))


def _determine_winner_loser(
    ev:        ConflictEvent,
    traj_map:  Dict[str, TrajectoryResult],
    locked:    Dict[FrozenSet, str],
) -> Tuple[str, str]:
    """
    Return (winner_id, loser_id) for the given ConflictEvent.

    The drone that traveled the greater arc-length during [t_start, t_end]
    is the winner.  Once decided, the outcome is stored in *locked* so the
    same pair always produces the same result across replan rounds.
    """
    pair_key = frozenset({ev.uav_a, ev.uav_b})

    if pair_key in locked:
        loser  = locked[pair_key]
        winner = ev.uav_b if loser == ev.uav_a else ev.uav_a
        return winner, loser

    len_a = _path_length_in_window(traj_map[ev.uav_a], ev.t_start, ev.t_end)
    len_b = _path_length_in_window(traj_map[ev.uav_b], ev.t_start, ev.t_end)

    if len_a >= len_b:
        winner, loser = ev.uav_a, ev.uav_b
    else:
        winner, loser = ev.uav_b, ev.uav_a

    locked[pair_key] = loser
    return winner, loser


def _find_pre_violation_segment(
    time_allocation: np.ndarray,
    t_violation_start: float,
) -> int:
    """
    Return the index of the STO segment that contains *t_violation_start*.

    This is the segment whose duration will be increased by Δt, causing
    the drone to arrive at the conflict zone later.

    Parameters
    ----------
    time_allocation : (n_segments,) per-segment durations [s]
    t_violation_start : float

    Returns
    -------
    int in [0, n_segments - 1]
    """
    t_cumsum = np.cumsum(time_allocation)
    # searchsorted 'left' → first index k where t_cumsum[k] >= t_violation_start
    # that is the segment whose end-time is at or after t_start, i.e. the
    # segment that contains t_violation_start.
    k = int(np.searchsorted(t_cumsum, t_violation_start, side="left"))
    return int(np.clip(k, 0, len(time_allocation) - 1))


def _distribute_delta_t(
    time_allocation: np.ndarray,
    k_pre:           int,
    delta_t:         float,
) -> np.ndarray:
    """
    Distribute ``delta_t`` across segments ``0 … k_pre`` (inclusive),
    proportionally to their current durations.

    The total added time equals ``delta_t`` exactly, so the UAV still
    arrives at the conflict zone ``delta_t`` seconds later.  Spreading the
    delay avoids disproportionately stretching short segments, which would
    otherwise create large velocity/acceleration spikes at their boundaries.

    Parameters
    ----------
    time_allocation : (n_segments,) current per-segment durations [s]
    k_pre           : index of the last pre-violation segment (inclusive)
    delta_t         : total seconds to add across segments 0 … k_pre

    Returns
    -------
    np.ndarray, shape (n_segments,) — modified time allocation
    """
    modified  = time_allocation.copy()
    pre_total = modified[:k_pre + 1].sum()
    for i in range(k_pre + 1):
        modified[i] += delta_t * (modified[i] / pre_total)
    return modified


def _earliest_event_per_pair(
    events: List[ConflictEvent],
) -> List[ConflictEvent]:
    """
    From a list of ConflictEvents, keep at most one per unordered UAV pair:
    the event with the smallest t_start.  Returns events sorted by t_start.
    """
    seen: Dict[FrozenSet, ConflictEvent] = {}
    for ev in events:
        key = frozenset({ev.uav_a, ev.uav_b})
        if key not in seen or ev.t_start < seen[key].t_start:
            seen[key] = ev
    return sorted(seen.values(), key=lambda e: e.t_start)


# ─────────────────────────────────────────────────────────────────────────────
# Main resolver
# ─────────────────────────────────────────────────────────────────────────────

def resolve_conflicts(
    fleet,
    trajectories:    List[TrajectoryResult],
    corridors:       list,
    plan_results:    list,
    conflict_report: ConflictReport,
    cd_cfg:          ConflictDetectorConfig,
    sto_cfg:         STOConfig,
    replan_cfg:      ReplanConfig,
    verbose:         bool = True,
) -> Tuple[List[TrajectoryResult], ConflictReport]:
    """
    Resolve inter-UAV trajectory conflicts via winner-locked temporal delay.

    For each violation the drone that traveled *less* arc-length through the
    conflict window (the "loser") has Δt distributed proportionally across its
    pre-conflict segments (0 … k_pre) and those segment durations are then
    **frozen** during the subsequent STO re-run.

    This guarantees two things simultaneously:
    - **Exact delay preservation**: tau[0..k_pre] are locked — the optimizer
      cannot reclaim the injected delay, so the UAV always arrives at the
      conflict zone Δt later.
    - **Corridor / kinematic compliance**: tau[k_pre+1..end] remain free;
      STO can adjust their timing to satisfy the soft corridor, velocity,
      and acceleration constraints that may have been disturbed by the
      changed boundary velocities at waypoint k_pre.

    Parameters
    ----------
    fleet            : Fleet
    trajectories     : list[TrajectoryResult]  — initial trajectories
    corridors        : list[CorridorResult]
    plan_results     : list[PlanResult]
    conflict_report  : ConflictReport          — initial detection result
    cd_cfg           : ConflictDetectorConfig
    sto_cfg          : STOConfig               — base STO hyper-parameters
    replan_cfg       : ReplanConfig
    verbose          : bool

    Returns
    -------
    (updated_trajectories, final_conflict_report)
    """
    if not conflict_report.has_conflicts:
        if verbose:
            print("No violations detected — replanning not needed.")
        return trajectories, conflict_report

    traj_map = {tr.uav_id: tr for tr in trajectories}
    plan_map = {pr.uav_id: pr for pr in plan_results}
    corr_map = {cr.uav_id: cr for cr in corridors}

    current_report  = conflict_report
    locked_decisions: Dict[FrozenSet, str] = {}

    for round_idx in range(replan_cfg.max_replan_rounds):
        n_events = len(current_report.events)
        if verbose:
            print(f"\n{'═' * 70}")
            print(f"  REPLAN ROUND {round_idx + 1}/{replan_cfg.max_replan_rounds}"
                  f"  ({n_events} violation(s) detected)")
            print(f"{'═' * 70}")

        # ── 1. One event per pair, sorted by earliest t_start ────────────────
        events_to_process = _earliest_event_per_pair(current_report.events)

        if verbose:
            print(f"\n  Processing {len(events_to_process)} unique pair(s)"
                  f" (earliest violation first):\n")

        # ── 2. Sequential processing ─────────────────────────────────────────
        for ev in events_to_process:
            winner, loser = _determine_winner_loser(ev, traj_map, locked_decisions)

            violation_duration = ev.t_end - ev.t_start
            delta_t = max(
                violation_duration * replan_cfg.delta_t_scale,
                replan_cfg.min_delta_t,
            )

            loser_traj = traj_map[loser]
            k_pre = _find_pre_violation_segment(
                loser_traj.time_allocation, ev.t_start
            )

            # Distribute delta_t proportionally across segments 0..k_pre
            time_init_mod = _distribute_delta_t(
                loser_traj.time_allocation, k_pre, delta_t
            )

            len_winner = _path_length_in_window(
                traj_map[winner], ev.t_start, ev.t_end
            )
            len_loser  = _path_length_in_window(
                traj_map[loser],  ev.t_start, ev.t_end
            )

            if verbose:
                print(f"  Pair  {ev.uav_a} <-> {ev.uav_b}")
                print(f"    violation window : [{ev.t_start:.2f} s → {ev.t_end:.2f} s]"
                      f"  (duration = {violation_duration:.2f} s)")
                print(f"    path length      : {ev.uav_a} = {len_winner if winner == ev.uav_a else len_loser:.2f} m"
                      f"  |  {ev.uav_b} = {len_loser if loser == ev.uav_b else len_winner:.2f} m")
                print(f"    winner (faster)  : {winner}  →  trajectory unchanged")
                print(f"    loser  (slower)  : {loser}   →  "
                      f"Δt={delta_t:.2f} s distributed across segs [0..{k_pre}]  "
                      f"({violation_duration:.2f} × {replan_cfg.delta_t_scale}"
                      f" = {violation_duration * replan_cfg.delta_t_scale:.2f},"
                      f" min={replan_cfg.min_delta_t})")
                for i in range(k_pre + 1):
                    old_ti = loser_traj.time_allocation[i]
                    new_ti = time_init_mod[i]
                    print(f"      seg {i}: {old_ti:.3f} s → {new_ti:.3f} s"
                          f"  (+{new_ti - old_ti:.3f} s)")

            uav = next(u for u in fleet.uavs if u.id == loser)

            new_tr = replan_single_trajectory(
                uav                     = uav,
                plan_result             = plan_map[loser],
                corridor_result         = corr_map[loser],
                sto_cfg                 = sto_cfg,
                frozen_trajectories     = [],
                lambda_sep              = 0.0,
                lambda_time_override    = replan_cfg.lambda_time_replan,
                replan_max_iter         = replan_cfg.replan_max_iter,
                temporal_only           = True,
                time_init_override      = time_init_mod,
                frozen_segment_indices  = list(range(k_pre + 1)),
                verbose                 = verbose,
            )

            if verbose:
                old_T = loser_traj.total_time
                new_T = new_tr.total_time
                print(f"    ✓  {loser}  T: {old_T:.2f} s → {new_T:.2f} s"
                      f"  (Δ = {new_T - old_T:+.2f} s)"
                      f"  jerk = {new_tr.jerk_cost:.4f}\n")

            traj_map[loser] = new_tr

        # ── 3. Re-detect conflicts ────────────────────────────────────────────
        updated        = [traj_map[u.id] for u in fleet.uavs]
        current_report = detect_conflicts(fleet, updated, corridors, cd_cfg)
        n_remaining    = len(current_report.events)

        if verbose:
            print(f"{'─' * 70}")
            print(f"  Re-check: {n_remaining} violation(s) remaining after round {round_idx + 1}")
            print(f"{'─' * 70}")

        if not current_report.has_conflicts:
            if verbose:
                print(f"\n  All violations resolved in {round_idx + 1} round(s).")
            break
    else:
        if current_report.has_conflicts and verbose:
            print(f"\n  WARNING: {len(current_report.events)} violation(s) remain "
                  f"after {replan_cfg.max_replan_rounds} round(s).")

    final_trajectories = [traj_map[u.id] for u in fleet.uavs]
    return final_trajectories, current_report
