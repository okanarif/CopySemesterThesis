"""Per-UAV STO planning pipeline.

Wraps the single-UAV flow (A* -> LOS prune -> pydecomp -> STOPlanner)
into a reusable function that can be called for each UAV in the swarm.
"""

from __future__ import annotations

import sys
import os
import numpy as np

from .uav import UAV
from .environment import Environment

# Ensure the STO directory is importable so we can reach sto_planner / utils
_sto_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "STO")
if _sto_dir not in sys.path:
    sys.path.insert(0, _sto_dir)

from sto_planner import STOPlanner  # type: ignore


def plan_single_uav(
    env: Environment,
    uav: UAV,
    *,
    v_max: float = 2.0,
    a_max: float = 1.0,
    lambda_jerk: float = 10.0,
    lambda_time: float = 0.05,
    lambda_vel: float = 120.0,
    lambda_acc: float = 5.0,
    lambda_corridor: float = 1000.0,
    corridor_cost_type: str = "l2",
    learn_rate: float = 0.05,
    max_iter: int = 50,
    adaptive_weights: bool = True,
    weight_phases: int = 3,
    weight_scale_factor: float = 5.0,
    verbose: bool = False,
    n_result_samples: int = 200,
) -> dict:
    """Run the full single-UAV planning pipeline and store results on *uav*.

    1. ``env.plan_path`` (A* -> LOS -> pydecomp)
    2. Initialise ``STOPlanner`` with corridor constraints
    3. ``planner.solve(...)`` + ``planner.get_results()``
    4. Store everything back into ``uav``

    Returns the STO results dict for convenience.
    """

    # --- Step 1: path planning + corridor decomposition ---
    path_np, A_list, b_list, path_indices = env.plan_path(uav.start, uav.goal)

    uav.path_indices = path_indices
    uav.path_np = path_np
    uav.A_list = A_list
    uav.b_list = b_list

    # --- Step 2: derive STO initialisation ---
    n_segments = len(A_list)
    waypoints_init = path_np[1:-1]

    segment_lengths = []
    segment_lengths.append(np.linalg.norm(waypoints_init[0] - path_np[0]))
    for i in range(len(waypoints_init) - 1):
        segment_lengths.append(
            np.linalg.norm(waypoints_init[i + 1] - waypoints_init[i])
        )
    segment_lengths.append(np.linalg.norm(path_np[-1] - waypoints_init[-1]))

    time_init = [max(length / v_max, 0.1) for length in segment_lengths]

    pos_init = path_np[0]
    pos_final = path_np[-1]
    vel_init = np.zeros(3)
    acc_init = np.zeros(3)
    vel_final = np.zeros(3)
    acc_final = np.zeros(3)

    # --- Step 3: create and run STO ---
    planner = STOPlanner(
        n_segments=n_segments,
        A_list=A_list,
        b_list=b_list,
        waypoints_init=waypoints_init,
        time_init=time_init,
        v_max=v_max,
        a_max=a_max,
        lambda_jerk=lambda_jerk,
        lambda_time=lambda_time,
        lambda_vel=lambda_vel,
        lambda_acc=lambda_acc,
        lambda_corridor=lambda_corridor,
        corridor_cost_type=corridor_cost_type,
        learn_rate=learn_rate,
        max_iter=max_iter,
        verbose=verbose,
        adaptive_weights=adaptive_weights,
        weight_phases=weight_phases,
        weight_scale_factor=weight_scale_factor,
    )

    planner.solve(
        pos_init=pos_init,
        vel_init=vel_init,
        acc_init=acc_init,
        pos_final=pos_final,
        vel_final=vel_final,
        acc_final=acc_final,
    )

    results = planner.get_results(n_samples=n_result_samples)
    uav.sto_results = results
    return results
