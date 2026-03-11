"""Centralized multi-UAV swarm planner.

Orchestrates per-UAV planning, conflict detection, and resolution.
"""

from __future__ import annotations

import time as time_module
from typing import Dict, List, Optional

import numpy as np

from .uav import UAV
from .environment import Environment
from .planner import plan_single_uav
from .conflict import (
    Conflict,
    detect_conflicts,
    resolve_conflicts_time_shift,
    compute_min_distances,
)


class SwarmPlanner:
    """Plan collision-free trajectories for a fleet of UAVs.

    Usage::

        env = Environment()
        uavs = [UAV("A", start_a, goal_a), UAV("B", start_b, goal_b)]
        sp = SwarmPlanner(env, uavs)
        results = sp.plan_all()
    """

    def __init__(
        self,
        env: Environment,
        uavs: List[UAV],
        sto_params: Optional[dict] = None,
        conflict_dt: float = 0.1,
        safety_margin: float = 0.5,
        max_resolve_rounds: int = 20,
        verbose: bool = True,
    ):
        self.env = env
        self.uavs = uavs
        self.sto_params = sto_params or {}
        self.conflict_dt = conflict_dt
        self.safety_margin = safety_margin
        self.max_resolve_rounds = max_resolve_rounds
        self.verbose = verbose

    def plan_all(self) -> dict:
        """Run the full pipeline: plan -> detect -> resolve.

        Returns
        -------
        dict with keys:
            uavs : list of UAV (with populated sto_results and time_offset)
            conflicts_initial : list of Conflict before resolution
            conflicts_remaining : list of Conflict after resolution
            planning_time : float (total wall-clock seconds)
            per_uav_times : dict mapping uav_id -> planning seconds
            min_distance_profile : (times, pair_dists) arrays
        """
        wall_start = time_module.time()
        per_uav_times: Dict[str, float] = {}

        # --- 1. Plan each UAV independently ---
        if self.verbose:
            print("=" * 60)
            print("SWARM PLANNER")
            print("=" * 60)
            print(f"Fleet size: {len(self.uavs)} UAVs")
            print(f"Safety margin: {self.safety_margin} m")
            print()

        for uav in self.uavs:
            if self.verbose:
                print(f"[plan] UAV {uav.uav_id}: {uav.start} -> {uav.goal}")
            t0 = time_module.time()
            params = dict(self.sto_params)
            params.setdefault("verbose", False)
            plan_single_uav(self.env, uav, **params)
            elapsed = time_module.time() - t0
            per_uav_times[uav.uav_id] = elapsed
            if self.verbose:
                print(
                    f"       done in {elapsed:.2f}s  "
                    f"(T={uav.total_time():.2f}s, "
                    f"segments={len(uav.A_list)})"
                )

        # --- 2. Detect conflicts ---
        conflicts_initial = detect_conflicts(
            self.uavs, dt=self.conflict_dt, safety_margin=self.safety_margin
        )
        if self.verbose:
            print(f"\n[detect] {len(conflicts_initial)} initial conflict(s)")
            for c in conflicts_initial:
                print(
                    f"   {c.uav_i_id} <-> {c.uav_j_id}  "
                    f"t=[{c.t_start:.2f}, {c.t_end:.2f}]  "
                    f"min_d={c.min_distance:.3f}"
                )

        # --- 3. Resolve conflicts ---
        conflicts_remaining = resolve_conflicts_time_shift(
            self.uavs,
            dt=self.conflict_dt,
            safety_margin=self.safety_margin,
            max_rounds=self.max_resolve_rounds,
            verbose=self.verbose,
        )

        if self.verbose:
            if conflicts_remaining:
                print(
                    f"\n[resolve] WARNING: {len(conflicts_remaining)} "
                    f"conflict(s) remain after {self.max_resolve_rounds} rounds"
                )
            else:
                print("\n[resolve] All conflicts resolved!")
            for u in self.uavs:
                print(
                    f"   UAV {u.uav_id}: offset={u.time_offset:.2f}s, "
                    f"T={u.total_time():.2f}s, "
                    f"end={u.time_offset + u.total_time():.2f}s"
                )

        # --- 4. Compute final distance profile ---
        dist_times, pair_dists = compute_min_distances(
            self.uavs, dt=self.conflict_dt
        )

        wall_time = time_module.time() - wall_start
        if self.verbose:
            print(f"\nTotal planning time: {wall_time:.2f}s")
            print("=" * 60)

        return {
            "uavs": self.uavs,
            "conflicts_initial": conflicts_initial,
            "conflicts_remaining": conflicts_remaining,
            "planning_time": wall_time,
            "per_uav_times": per_uav_times,
            "min_distance_profile": (dist_times, pair_dists),
        }
