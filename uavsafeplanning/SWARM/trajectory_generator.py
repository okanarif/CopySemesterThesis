"""
Trajectory generator — wraps STO_Swarm_Planner for multi-UAV use.

For each UAV in the fleet:
  1. Build initial waypoints from the pruned A* path  (path[1:-1]).
  2. Build initial time allocation: segment_length / v_max per segment.
  3. Initialise STO_Swarm_Planner with the UAV's safe corridors and dynamic limits.
  4. Run L-BFGS optimisation (adaptive-weight STO).
  5. Return the sampled trajectory (position, velocity, acceleration).

Public API
----------
    from trajectory_generator import generate_trajectories, TrajectoryResult, STOConfig

    sto_cfg = STOConfig.from_yaml("configs/planner.yaml")
    traj_results = generate_trajectories(fleet, plan_results, corridor_results,
                                         sto_cfg, verbose=True)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
import yaml

# ── path bootstrap ─────────────────────────────────────────────────────────────
_swarm_dir   = os.path.dirname(os.path.abspath(__file__))
_uavsafe_dir = os.path.dirname(_swarm_dir)
_sto_dir     = os.path.join(_uavsafe_dir, "STO_Swarm")

for _p in (_swarm_dir, _uavsafe_dir, _sto_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sto_swarm_planner import STO_Swarm_Planner
from uav         import Fleet
from logger      import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class STOConfig:
    """STO optimiser hyper-parameters loaded from the sto: block of planner.yaml."""
    max_iter:            int   = 50
    lambda_jerk:         float = 10.0
    lambda_time:         float = 0.05
    lambda_vel:          float = 120.0
    lambda_acc:          float = 5.0
    lambda_corridor:     float = 1000.0
    corridor_cost_type:  str   = "l2"
    learn_rate:          float = 0.05
    adaptive_weights:    bool  = True
    weight_phases:       int   = 3
    weight_scale_factor: float = 5.0
    n_samples:           int   = 300

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "STOConfig":
        """Load and validate STO config from a YAML file."""
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}

        sto = raw.get("sto", {})

        cost_type = str(sto.get("corridor_cost_type", "l2")).lower()
        if cost_type not in ("l1", "l2", "log"):
            raise ValueError(
                f"planner.yaml: sto.corridor_cost_type must be l1 | l2 | log "
                f"(got '{cost_type}')"
            )

        return cls(
            max_iter            = int(sto.get("max_iter",            50)),
            lambda_jerk         = float(sto.get("lambda_jerk",       10.0)),
            lambda_time         = float(sto.get("lambda_time",        0.05)),
            lambda_vel          = float(sto.get("lambda_vel",         120.0)),
            lambda_acc          = float(sto.get("lambda_acc",         5.0)),
            lambda_corridor     = float(sto.get("lambda_corridor",    1000.0)),
            corridor_cost_type  = cost_type,
            learn_rate          = float(sto.get("learn_rate",         0.05)),
            adaptive_weights    = bool(sto.get("adaptive_weights",    True)),
            weight_phases       = int(sto.get("weight_phases",        3)),
            weight_scale_factor = float(sto.get("weight_scale_factor", 5.0)),
            n_samples           = int(sto.get("n_samples",            300)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryResult:
    """STO-optimised trajectory for one UAV."""
    uav_id:          str
    pos:             np.ndarray   # (n_samples, 3) — world coords [m]
    vel:             np.ndarray   # (n_samples, 3) — [m/s]
    acc:             np.ndarray   # (n_samples, 3) — [m/s²]
    vel_norm:        np.ndarray   # (n_samples,)   — speed scalar [m/s]
    t_eval:          np.ndarray   # (n_samples,)   — time stamps [s]
    total_time:      float        # total trajectory duration [s]
    jerk_cost:       float        # final jerk-energy cost
    violations:      dict         # {"vel": float, "acc": float, "corridor": float}
    runtime:         float        # wall-clock optimisation time [s]
    time_allocation: np.ndarray   # (n_segments,) — per-segment durations [s]


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_trajectories(
    fleet:            Fleet,
    plan_results:     list,    # list[PlanResult]
    corridor_results: list,    # list[CorridorResult]
    sto_cfg:          STOConfig,
    verbose:          bool = True,
) -> list[TrajectoryResult]:
    """
    Run STO for every UAV in the fleet.

    Parameters
    ----------
    fleet            : Fleet
    plan_results     : list[PlanResult]     — from global_planner.plan_paths()
    corridor_results : list[CorridorResult] — from safe_corridor.extract_corridors()
    sto_cfg          : STOConfig            — optimiser hyper-parameters
    verbose          : bool                 — print per-UAV optimisation log

    Returns
    -------
    list[TrajectoryResult]
        One entry per UAV, in the same order as ``fleet.uavs``.

    Raises
    ------
    RuntimeError  — if STO_Swarm_Planner fails for any UAV.
    """
    plan_map     = {pr.uav_id: pr for pr in plan_results}
    corridor_map = {cr.uav_id: cr for cr in corridor_results}

    results: list[TrajectoryResult] = []

    for uav in fleet.uavs:
        pr = plan_map[uav.id]
        cr = corridor_map[uav.id]

        # ── Initial waypoints & time allocation ──────────────────────────────
        path_world     = pr.path_world                  # (N, 3)
        waypoints_init = path_world[1:-1]               # (N-2, 3); empty if N == 2

        seg_lengths = np.linalg.norm(np.diff(path_world, axis=0), axis=1)  # (N-1,)
        time_init   = (seg_lengths / max(uav.v_max, 1e-6)).tolist()

        n_segments = len(cr.A_list)

        if verbose:
            print(f"\n{'─' * 62}")
            print(f"  STO  ▶  {uav.id}   "
                  f"({n_segments} segment{'s' if n_segments != 1 else ''},  "
                  f"{len(waypoints_init)} waypoint{'s' if len(waypoints_init) != 1 else ''})")
            print(f"{'─' * 62}")

        # ── STO_Swarm_Planner ─────────────────────────────────────────────────
        try:
            planner = STO_Swarm_Planner(
                n_segments          = n_segments,
                A_list              = cr.A_list,
                b_list              = cr.b_list,
                waypoints_init      = waypoints_init,
                time_init           = np.array(time_init),
                v_max               = uav.v_max,
                a_max               = uav.a_max,
                lambda_jerk         = sto_cfg.lambda_jerk,
                lambda_time         = sto_cfg.lambda_time,
                lambda_vel          = sto_cfg.lambda_vel,
                lambda_acc          = sto_cfg.lambda_acc,
                lambda_corridor     = sto_cfg.lambda_corridor,
                corridor_cost_type  = sto_cfg.corridor_cost_type,
                learn_rate          = sto_cfg.learn_rate,
                max_iter            = sto_cfg.max_iter,
                verbose             = verbose,
                adaptive_weights    = sto_cfg.adaptive_weights,
                weight_phases       = sto_cfg.weight_phases,
                weight_scale_factor = sto_cfg.weight_scale_factor,
            )

            planner.solve(
                pos_init  = uav.start,
                vel_init  = np.zeros(3),
                acc_init  = np.zeros(3),
                pos_final = uav.goal,
                vel_final = np.zeros(3),
                acc_final = np.zeros(3),
            )

            res = planner.get_results(n_samples=sto_cfg.n_samples)

        except Exception as exc:
            raise RuntimeError(
                f"STO optimisation failed for '{uav.id}': {exc}"
            ) from exc

        results.append(TrajectoryResult(
            uav_id          = uav.id,
            pos             = res["pos"],
            vel             = res["vel"],
            acc             = res["acc"],
            vel_norm        = res["vel_norm"],
            t_eval          = res["t_eval"],
            total_time      = float(res["total_time"]),
            jerk_cost       = float(res["jerk_cost"]),
            violations      = res["violations"],
            runtime         = float(res["runtime"]),
            time_allocation = res["time_allocation"],
        ))

        log.debug(
            f"{uav.id}:  T={res['total_time']:.1f} s  "
            f"jerk={res['jerk_cost']:.4f}  "
            f"viol_vel={res['violations']['vel']:.4f}  "
            f"viol_corr={res['violations']['corridor']:.6f}  "
            f"({res['runtime']:.1f} s)"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Single-UAV replan (temporal de-confliction)
# ─────────────────────────────────────────────────────────────────────────────

def replan_single_trajectory(
    uav,
    plan_result,
    corridor_result,
    sto_cfg:               STOConfig,
    frozen_trajectories:   list,
    lambda_sep:            float,
    lambda_time_override:  float | None      = None,
    replan_max_iter:       int   | None      = None,
    temporal_only:         bool              = True,
    time_init_override:    np.ndarray | None = None,
    verbose:               bool              = True,
) -> TrajectoryResult:
    """
    Re-run STO for a single UAV, optionally with a custom time initialisation.

    Parameters
    ----------
    uav              : UAVConfig
    plan_result      : PlanResult
    corridor_result  : CorridorResult
    sto_cfg          : STOConfig — base hyper-parameters
    frozen_trajectories : list[(t_eval_np, pos_np, safety_dist)]
    lambda_sep       : float — weight for the separation penalty (0 = disabled)
    lambda_time_override : float, optional — reduced lambda_time for replan
    replan_max_iter  : int, optional — override max_iter for replan
    temporal_only    : bool — freeze waypoints, only optimise tau
    time_init_override : np.ndarray, optional — per-segment time initialisation
        (n_segments,).  When provided, overrides the default seg_len/v_max
        initialisation.  Use this to inject a pre-violation delay.
    verbose          : bool

    Returns
    -------
    TrajectoryResult
    """
    pr = plan_result
    cr = corridor_result

    path_world     = pr.path_world
    waypoints_init = path_world[1:-1]
    seg_lengths    = np.linalg.norm(np.diff(path_world, axis=0), axis=1)
    n_segments     = len(cr.A_list)

    if time_init_override is not None:
        time_init = time_init_override.tolist()
    else:
        time_init = (seg_lengths / max(uav.v_max, 1e-6)).tolist()

    effective_time  = lambda_time_override if lambda_time_override is not None else sto_cfg.lambda_time
    effective_iter  = replan_max_iter if replan_max_iter is not None else sto_cfg.max_iter

    if verbose:
        print(f"\n{'─' * 62}")
        print(f"  REPLAN  ▶  {uav.id}   "
              f"({n_segments} segments,  temporal_only={temporal_only},  "
              f"λ_sep={lambda_sep:.0f})")
        print(f"{'─' * 62}")

    try:
        planner = STO_Swarm_Planner(
            n_segments          = n_segments,
            A_list              = cr.A_list,
            b_list              = cr.b_list,
            waypoints_init      = waypoints_init,
            time_init           = np.array(time_init),
            v_max               = uav.v_max,
            a_max               = uav.a_max,
            lambda_jerk         = sto_cfg.lambda_jerk,
            lambda_time         = effective_time,
            lambda_vel          = sto_cfg.lambda_vel,
            lambda_acc          = sto_cfg.lambda_acc,
            lambda_corridor     = sto_cfg.lambda_corridor,
            corridor_cost_type  = sto_cfg.corridor_cost_type,
            learn_rate          = sto_cfg.learn_rate,
            max_iter            = effective_iter,
            verbose             = verbose,
            adaptive_weights    = sto_cfg.adaptive_weights,
            weight_phases       = sto_cfg.weight_phases,
            weight_scale_factor = sto_cfg.weight_scale_factor,
            frozen_trajectories = frozen_trajectories,
            lambda_sep          = lambda_sep,
            temporal_only       = temporal_only,
        )

        planner.solve(
            pos_init  = uav.start,
            vel_init  = np.zeros(3),
            acc_init  = np.zeros(3),
            pos_final = uav.goal,
            vel_final = np.zeros(3),
            acc_final = np.zeros(3),
        )

        res = planner.get_results(n_samples=sto_cfg.n_samples)

    except Exception as exc:
        raise RuntimeError(
            f"STO replan failed for '{uav.id}': {exc}"
        ) from exc

    return TrajectoryResult(
        uav_id          = uav.id,
        pos             = res["pos"],
        vel             = res["vel"],
        acc             = res["acc"],
        vel_norm        = res["vel_norm"],
        t_eval          = res["t_eval"],
        total_time      = float(res["total_time"]),
        jerk_cost       = float(res["jerk_cost"]),
        violations      = res["violations"],
        runtime         = float(res["runtime"]),
        time_allocation = res["time_allocation"],
    )
