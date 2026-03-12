"""
Space-time conflict detection for multi-UAV swarm trajectories.

Detection strategy
------------------
Two complementary checks are run for every ordered pair of UAVs:

  1. Trajectory-proximity check (primary)
     Both trajectories are linearly interpolated onto a common time grid
     (step = dt_sample seconds).  Every contiguous window where the
     Euclidean separation drops below the combined safety distance
     (r_i + r_j + safety_margin) is reported as a ConflictEvent.

  2. Safe-corridor intersection analysis (structural)
     For each pair of corridor polytopes  {A_i x ≤ b_i}  and
     {A_j x ≤ b_j}  an LP feasibility test decides whether the two
     convex regions share any volume.  For spatially intersecting pairs
     the time windows during which each UAV occupies its respective
     polytope are derived from the sampled trajectory; overlapping
     windows yield a CorridorConflictZone flagged as a spatio-temporal
     conflict.

Public API
----------
    from conflict_detector import (
        ConflictDetectorConfig,
        ConflictEvent,
        CorridorConflictZone,
        ConflictReport,
        detect_conflicts,
        plot_conflict_report,
    )

    cfg    = ConflictDetectorConfig.from_yaml("configs/planner.yaml")
    report = detect_conflicts(fleet, trajectories, corridors, cfg)
    print(report.summary())
    plot_conflict_report(report, fleet, trajectories)
"""

from __future__ import annotations

import itertools
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.interpolate import interp1d
from scipy.optimize import linprog

# ── path bootstrap ─────────────────────────────────────────────────────────────
_swarm_dir   = os.path.dirname(os.path.abspath(__file__))
_uavsafe_dir = os.path.dirname(_swarm_dir)
for _p in (_swarm_dir, _uavsafe_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConflictDetectorConfig:
    """
    Parameters for the conflict detector, loaded from the
    ``conflict_detection:`` block of ``planner.yaml``.

    Attributes
    ----------
    safety_margin : float
        Extra separation buffer added on top of the two UAVs' physical
        radii  (r_i + r_j + safety_margin).  Set to 0 for a hard-sphere
        check; increase for a conservative separation guarantee  [m].
    dt_sample : float
        Time step used to build the common evaluation grid  [s].
        Smaller values increase temporal resolution at a proportional
        cost; 50 ms is a good default for this scenario.
    """
    safety_margin: float = 0.0
    dt_sample:     float = 0.05

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ConflictDetectorConfig":
        """
        Load the ``conflict_detection:`` block from *path* (planner.yaml).

        Missing keys fall back to dataclass defaults so the block is
        entirely optional in the YAML file.

        Raises
        ------
        ValueError
            If a field value fails a range check.
        """
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        cd = raw.get("conflict_detection", {})

        safety_margin = float(cd.get("safety_margin", 0.0))
        dt_sample     = float(cd.get("dt_sample",     0.05))

        if safety_margin < 0:
            raise ValueError(
                f"planner.yaml: conflict_detection.safety_margin must be >= 0 "
                f"(got {safety_margin})"
            )
        if dt_sample <= 0:
            raise ValueError(
                f"planner.yaml: conflict_detection.dt_sample must be > 0 "
                f"(got {dt_sample})"
            )

        return cls(safety_margin=safety_margin, dt_sample=dt_sample)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConflictEvent:
    """
    A contiguous time window where two UAVs violate their combined
    safety distance, detected via the trajectory-proximity check.

    Attributes
    ----------
    uav_a, uav_b : str
        Identifiers of the two UAVs involved.
    t_start, t_end : float
        First and last time sample inside the violation window  [s].
    min_dist : float
        Minimum Euclidean separation reached during the window  [m].
    t_min_dist : float
        Time at which the minimum separation occurs  [s].
    pos_a_at_min, pos_b_at_min : np.ndarray  shape (3,)
        World-frame positions of each UAV at the moment of closest
        approach  [m].
    safety_dist : float
        The threshold that was violated  (r_a + r_b + margin)  [m].
    """
    uav_a:        str
    uav_b:        str
    t_start:      float
    t_end:        float
    min_dist:     float
    t_min_dist:   float
    pos_a_at_min: np.ndarray
    pos_b_at_min: np.ndarray
    safety_dist:  float

    @property
    def duration(self) -> float:
        """Length of the violation window  [s]."""
        return self.t_end - self.t_start

    @property
    def separation_vector_at_min(self) -> np.ndarray:
        """Vector from UAV-B to UAV-A at the moment of closest approach."""
        return self.pos_a_at_min - self.pos_b_at_min


@dataclass
class CorridorConflictZone:
    """
    A pair of safe-corridor polytopes that overlap in space, together
    with the time windows during which each UAV occupies its polytope.

    Attributes
    ----------
    uav_a, uav_b : str
    poly_idx_a, poly_idx_b : int
        Indices into the respective ``A_list`` / ``b_list`` of each UAV.
    t_window_a, t_window_b : (float, float)
        ``(t_enter, t_exit)`` intervals extracted from trajectory samples.
    time_overlap : bool
        ``True`` when  [t_window_a]  and  [t_window_b]  overlap in time,
        meaning both UAVs are simultaneously in the shared volume.
    """
    uav_a:        str
    uav_b:        str
    poly_idx_a:   int
    poly_idx_b:   int
    t_window_a:   Tuple[float, float]
    t_window_b:   Tuple[float, float]
    time_overlap: bool


@dataclass
class ConflictReport:
    """
    Complete space-time conflict report for the fleet.

    Attributes
    ----------
    cfg : ConflictDetectorConfig
        The configuration used to produce this report.
    events : list[ConflictEvent]
        All trajectory-proximity violations found.
    corridor_zones : list[CorridorConflictZone]
        All spatially intersecting corridor polytope pairs.
    """
    cfg:            ConflictDetectorConfig
    events:         List[ConflictEvent]        = field(default_factory=list)
    corridor_zones: List[CorridorConflictZone] = field(default_factory=list)

    # ── quick queries ──────────────────────────────────────────────────────────

    @property
    def has_conflicts(self) -> bool:
        """True if any trajectory-proximity violation was detected."""
        return len(self.events) > 0

    @property
    def n_corridor_overlaps(self) -> int:
        """Number of corridor zones that overlap in both space and time."""
        return sum(1 for z in self.corridor_zones if z.time_overlap)

    def events_for_pair(self, id_a: str, id_b: str) -> List[ConflictEvent]:
        """Return all ConflictEvents for the unordered pair (id_a, id_b)."""
        pair = {id_a, id_b}
        return [ev for ev in self.events if {ev.uav_a, ev.uav_b} == pair]

    def corridor_zones_for_pair(
        self, id_a: str, id_b: str
    ) -> List[CorridorConflictZone]:
        """Return all CorridorConflictZones for the unordered pair (id_a, id_b)."""
        pair = {id_a, id_b}
        return [z for z in self.corridor_zones if {z.uav_a, z.uav_b} == pair]

    # ── human-readable report ──────────────────────────────────────────────────

    def summary(self) -> str:
        W   = 70
        bar = "─" * W
        lines = [
            bar,
            "  SPACE-TIME CONFLICT DETECTION REPORT",
            bar,
            f"  Safety margin (extra)   : {self.cfg.safety_margin:.3f} m"
            f"  (threshold = r_i + r_j + margin)",
            f"  Time-grid resolution    : {self.cfg.dt_sample * 1000:.0f} ms",
            bar,
        ]

        # ── trajectory events ──────────────────────────────────────────────
        if not self.events:
            lines.append("  OK  No trajectory separation violations detected.")
        else:
            lines.append(f"  !!  {len(self.events)} violation(s) detected:\n")
            for ev in self.events:
                lines += [
                    f"     +-- {ev.uav_a}  <->  {ev.uav_b}",
                    f"     |  safety threshold : {ev.safety_dist:.3f} m",
                    f"     |  window           : [{ev.t_start:.3f} s -> {ev.t_end:.3f} s]"
                    f"  (dt = {ev.duration:.3f} s)",
                    f"     |  closest approach : {ev.min_dist:.4f} m"
                    f"  at t = {ev.t_min_dist:.3f} s",
                    f"     |  pos A            : {ev.pos_a_at_min.round(2).tolist()}",
                    f"     +  pos B            : {ev.pos_b_at_min.round(2).tolist()}",
                    "",
                ]

        lines.append(bar)

        # ── corridor zones ─────────────────────────────────────────────────
        n_spatial  = len(self.corridor_zones)
        n_temporal = self.n_corridor_overlaps
        lines.append(
            f"  Corridor polytope spatial intersections : {n_spatial}"
        )
        lines.append(
            f"  -> with simultaneous temporal overlap   : {n_temporal}"
        )

        if self.corridor_zones:
            lines.append("")
            for z in self.corridor_zones:
                mark = "  [!]" if z.time_overlap else "  [ ]"
                tag  = "  <- SPATIO-TEMPORAL CONFLICT" if z.time_overlap else ""
                lines.append(
                    f"{mark} [{z.uav_a} P{z.poly_idx_a}] x [{z.uav_b} P{z.poly_idx_b}]"
                    f"   A:[{z.t_window_a[0]:.2f}-{z.t_window_a[1]:.2f} s]"
                    f"   B:[{z.t_window_b[0]:.2f}-{z.t_window_b[1]:.2f} s]"
                    f"{tag}"
                )

        lines.append(bar)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _interp_traj(tr, t_grid: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate a TrajectoryResult's position onto *t_grid*.

    Points outside the UAV's flight window are clamped to its
    start / end position.

    Parameters
    ----------
    tr     : TrajectoryResult — must expose ``.t_eval`` and ``.pos``.
    t_grid : (N,) common time axis.

    Returns
    -------
    np.ndarray  shape (N, 3)
    """
    t   = tr.t_eval   # (n_samples,)
    pos = tr.pos      # (n_samples, 3)
    out = np.empty((len(t_grid), 3), dtype=np.float64)
    for k in range(3):
        f = interp1d(
            t, pos[:, k],
            kind="linear",
            bounds_error=False,
            fill_value=(pos[0, k], pos[-1, k]),
        )
        out[:, k] = f(t_grid)
    return out


def _polytope_time_window(
    A:      np.ndarray,
    b:      np.ndarray,
    pos:    np.ndarray,
    t_eval: np.ndarray,
    tol:    float = 1e-6,
) -> Optional[Tuple[float, float]]:
    """
    Return the time span ``(t_enter, t_exit)`` during which *pos* samples
    satisfy  ``A @ x <= b + tol``  (i.e. lie inside the polytope).

    Parameters
    ----------
    A      : (n_c, 3) half-space normal matrix.
    b      : (n_c,)   offset vector.
    pos    : (n, 3)   sampled trajectory positions.
    t_eval : (n,)     corresponding time stamps.
    tol    : numerical tolerance to handle boundary samples.

    Returns
    -------
    (t_enter, t_exit) or None if no sample is inside the polytope.
    """
    # b may arrive as (n_c, 1) from pydecomp — flatten to (n_c,)
    b = np.asarray(b, dtype=np.float64).ravel()
    # (n, n_c): each row holds  A @ pos[i]
    inside = np.all(pos @ A.T <= b[None, :] + tol, axis=1)
    idx    = np.where(inside)[0]
    if len(idx) == 0:
        return None
    return (float(t_eval[idx[0]]), float(t_eval[idx[-1]]))


def _polytopes_intersect(
    A1: np.ndarray,
    b1: np.ndarray,
    A2: np.ndarray,
    b2: np.ndarray,
) -> bool:
    """
    Return ``True`` if  ``{A1 x <= b1}``  and  ``{A2 x <= b2}``  share a
    common interior point.

    Implemented as a pure LP feasibility check:
        minimise   0
        subject to [A1; A2] x <= [b1; b2]

    Uses the HiGHS interior-point solver via ``scipy.optimize.linprog``.
    """
    res = linprog(
        np.zeros(3),
        A_ub=np.vstack([A1, A2]),
        b_ub=np.concatenate([b1.ravel(), b2.ravel()]),
        bounds=[(None, None)] * 3,
        method="highs",
        options={"disp": False},
    )
    return res.status == 0   # 0 = optimal (feasible)


def _find_conflict_events(
    id_a:        str,
    id_b:        str,
    pos_a:       np.ndarray,
    pos_b:       np.ndarray,
    t_grid:      np.ndarray,
    safety_dist: float,
) -> List[ConflictEvent]:
    """
    Scan the pairwise distance profile and return one ``ConflictEvent``
    per contiguous window where  ``||pos_a - pos_b|| < safety_dist``.

    Parameters
    ----------
    id_a, id_b   : UAV identifiers.
    pos_a, pos_b : (N, 3) positions on the common time grid.
    t_grid       : (N,) common time axis.
    safety_dist  : separation threshold  [m].
    """
    dist = np.linalg.norm(pos_a - pos_b, axis=1)
    mask = dist < safety_dist

    if not np.any(mask):
        return []

    # Detect rising / falling edges via 1-D convolution of the boolean mask
    padded  = np.concatenate([[False], mask, [False]])
    changes = np.diff(padded.astype(np.int8))
    starts  = np.where(changes ==  1)[0]   # first index inside each window
    ends    = np.where(changes == -1)[0]   # first index after each window

    events: List[ConflictEvent] = []
    for s, e in zip(starts, ends):
        seg   = dist[s:e]
        i_min = s + int(np.argmin(seg))
        events.append(ConflictEvent(
            uav_a        = id_a,
            uav_b        = id_b,
            t_start      = float(t_grid[s]),
            t_end        = float(t_grid[e - 1]),
            min_dist     = float(dist[i_min]),
            t_min_dist   = float(t_grid[i_min]),
            pos_a_at_min = pos_a[i_min].copy(),
            pos_b_at_min = pos_b[i_min].copy(),
            safety_dist  = safety_dist,
        ))
    return events


def _check_corridor_pair(
    id_a:  str,
    id_b:  str,
    cr_a,
    cr_b,
    tr_a,
    tr_b,
) -> List[CorridorConflictZone]:
    """
    Test all polytope pairs from two UAVs' safe corridors for spatial
    and temporal intersection, returning a list of CorridorConflictZones.

    Parameters
    ----------
    id_a, id_b : UAV identifiers.
    cr_a, cr_b : CorridorResult objects (must expose A_list, b_list, and
                 the owning trajectory's t_eval is accessed via tr_*).
    tr_a, tr_b : TrajectoryResult objects (must expose .pos and .t_eval).
    """
    zones: List[CorridorConflictZone] = []

    for ki, (A_a, b_a) in enumerate(zip(cr_a.A_list, cr_a.b_list)):
        A_a_np = np.asarray(A_a, dtype=np.float64)
        b_a_np = np.asarray(b_a, dtype=np.float64).ravel()

        win_a = _polytope_time_window(A_a_np, b_a_np, tr_a.pos, tr_a.t_eval)
        if win_a is None:
            continue

        for kj, (A_b, b_b) in enumerate(zip(cr_b.A_list, cr_b.b_list)):
            A_b_np = np.asarray(A_b, dtype=np.float64)
            b_b_np = np.asarray(b_b, dtype=np.float64).ravel()

            if not _polytopes_intersect(A_a_np, b_a_np, A_b_np, b_b_np):
                continue

            win_b = _polytope_time_window(A_b_np, b_b_np, tr_b.pos, tr_b.t_eval)
            if win_b is None:
                continue

            ta0, ta1 = win_a
            tb0, tb1 = win_b
            t_overlap = (ta0 <= tb1) and (tb0 <= ta1)

            zones.append(CorridorConflictZone(
                uav_a        = id_a,
                uav_b        = id_b,
                poly_idx_a   = ki,
                poly_idx_b   = kj,
                t_window_a   = win_a,
                t_window_b   = win_b,
                time_overlap = t_overlap,
            ))

    return zones


# ─────────────────────────────────────────────────────────────────────────────
# Main detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_conflicts(
    fleet,
    trajectories:  list,
    corridors:     list,
    cfg:           ConflictDetectorConfig,
) -> ConflictReport:
    """
    Detect space-time conflicts for every UAV pair in the fleet.

    Parameters
    ----------
    fleet        : Fleet
    trajectories : list[TrajectoryResult]  — from ``generate_trajectories()``
    corridors    : list[CorridorResult]    — from ``extract_corridors()``
    cfg          : ConflictDetectorConfig  — loaded from planner.yaml

    Returns
    -------
    ConflictReport
    """
    traj_map   = {tr.uav_id: tr for tr in trajectories}
    corr_map   = {cr.uav_id: cr for cr in corridors}
    radius_map = {u.id: u.radius for u in fleet.uavs}
    uav_ids    = [u.id for u in fleet.uavs]

    t_max  = max(tr.total_time for tr in trajectories)
    t_grid = np.arange(0.0, t_max + cfg.dt_sample, cfg.dt_sample)

    log.debug(
        f"Conflict detection: {len(uav_ids)} UAVs  "
        f"T_max={t_max:.1f} s  grid_pts={len(t_grid)}  "
        f"margin={cfg.safety_margin:.3f} m"
    )

    report = ConflictReport(cfg=cfg)

    for id_a, id_b in itertools.combinations(uav_ids, 2):
        tr_a        = traj_map[id_a]
        tr_b        = traj_map[id_b]
        cr_a        = corr_map[id_a]
        cr_b        = corr_map[id_b]
        safety_dist = radius_map[id_a] + radius_map[id_b] + cfg.safety_margin

        # 1 — trajectory proximity -----------------------------------------
        pos_a = _interp_traj(tr_a, t_grid)
        pos_b = _interp_traj(tr_b, t_grid)

        events = _find_conflict_events(
            id_a, id_b, pos_a, pos_b, t_grid, safety_dist
        )
        report.events.extend(events)

        log.debug(
            f"  [{id_a} <-> {id_b}]  threshold={safety_dist:.3f} m  "
            f"events={len(events)}"
        )

        # 2 — corridor intersection analysis --------------------------------
        zones = _check_corridor_pair(id_a, id_b, cr_a, cr_b, tr_a, tr_b)
        report.corridor_zones.extend(zones)

        log.debug(
            f"  [{id_a} <-> {id_b}]  corridor zones={len(zones)}"
            f"  temporal_overlaps={sum(z.time_overlap for z in zones)}"
        )

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_conflict_report(
    report:       ConflictReport,
    fleet,
    trajectories: list,
) -> None:
    """
    Plot pairwise separation distance vs. time for every UAV pair.

    Visual encoding
    ---------------
    - Blue line         : Euclidean separation  ||pos_a(t) - pos_b(t)||
    - Red dashed line   : safety threshold  (r_i + r_j + margin)
    - Red shading       : trajectory-proximity violation windows
    - Pink shading      : corridor spatio-temporal overlap windows
    - Red triangle (v)  : moment of closest approach per violation
    - Dotted verticals  : each UAV's landing (end of trajectory) time

    Parameters
    ----------
    report       : ConflictReport — output of ``detect_conflicts()``.
    fleet        : Fleet
    trajectories : list[TrajectoryResult]
    """
    traj_map   = {tr.uav_id: tr for tr in trajectories}
    radius_map = {u.id: u.radius for u in fleet.uavs}
    color_map  = {u.id: u.color  for u in fleet.uavs}
    uav_ids    = [u.id for u in fleet.uavs]
    pairs      = list(itertools.combinations(uav_ids, 2))

    t_max  = max(tr.total_time for tr in trajectories)
    t_grid = np.arange(0.0, t_max + report.cfg.dt_sample, report.cfg.dt_sample)

    fig, axes = plt.subplots(
        len(pairs), 1,
        figsize=(13, 4 * len(pairs)),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(
        "Space-Time Conflict Analysis — Pairwise Separation",
        fontsize=14, fontweight="bold",
    )

    for ax, (id_a, id_b) in zip(axes[:, 0], pairs):
        tr_a        = traj_map[id_a]
        tr_b        = traj_map[id_b]
        safety_dist = radius_map[id_a] + radius_map[id_b] + report.cfg.safety_margin

        pos_a = _interp_traj(tr_a, t_grid)
        pos_b = _interp_traj(tr_b, t_grid)
        dist  = np.linalg.norm(pos_a - pos_b, axis=1)

        _draw_pair(
            ax, id_a, id_b,
            dist, t_grid, safety_dist,
            report.events_for_pair(id_a, id_b),
            report.corridor_zones_for_pair(id_a, id_b),
            tr_a.total_time, tr_b.total_time,
            color_map[id_a], color_map[id_b],
            radius_map[id_a], radius_map[id_b],
        )

    plt.show()


def _draw_pair(
    ax,
    id_a:          str,
    id_b:          str,
    dist:          np.ndarray,
    t_grid:        np.ndarray,
    safety_dist:   float,
    events:        List[ConflictEvent],
    zones:         List[CorridorConflictZone],
    t_land_a:      float,
    t_land_b:      float,
    color_a:       str,
    color_b:       str,
    radius_a:      float,
    radius_b:      float,
) -> None:
    """Render one subplot for the pair (id_a, id_b)."""

    # ── corridor temporal-overlap bands (drawn first, behind everything) ──
    first_zone = True
    for z in zones:
        if not z.time_overlap:
            continue
        ta0, ta1 = z.t_window_a
        tb0, tb1 = z.t_window_b
        ax.axvspan(
            max(ta0, tb0), min(ta1, tb1),
            alpha=0.10, color="red",
            label=f"corridor overlap P{z.poly_idx_a}/P{z.poly_idx_b}"
                  if first_zone else "",
        )
        first_zone = False

    # ── distance curve ────────────────────────────────────────────────────
    ax.plot(t_grid, dist, lw=1.8, color="steelblue",
            label=f"||{id_a} - {id_b}||")

    # ── safety threshold ──────────────────────────────────────────────────
    ax.axhline(safety_dist, color="crimson", lw=1.5, ls="--",
               label=f"safety threshold  {safety_dist:.2f} m")

    # ── trajectory violation windows and closest-approach markers ─────────
    for i, ev in enumerate(events):
        ax.axvspan(ev.t_start, ev.t_end, alpha=0.25, color="crimson",
                   label="separation violation" if i == 0 else "")
        ax.plot(ev.t_min_dist, ev.min_dist, "rv", ms=9, zorder=5,
                label=f"closest {ev.min_dist:.3f} m @ {ev.t_min_dist:.2f} s")

    # ── landing-time markers ──────────────────────────────────────────────
    ax.axvline(t_land_a, color=color_a, lw=1.2, ls=":", alpha=0.8,
               label=f"{id_a} lands @ {t_land_a:.1f} s")
    ax.axvline(t_land_b, color=color_b, lw=1.2, ls=":", alpha=0.8,
               label=f"{id_b} lands @ {t_land_b:.1f} s")

    ax.set_xlim(0, t_grid[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time  [s]")
    ax.set_ylabel("Separation  [m]")
    ax.set_title(
        f"{id_a}  <->  {id_b}   "
        f"(r_a={radius_a:.2f} m,  r_b={radius_b:.2f} m,  "
        f"threshold={safety_dist:.2f} m)"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.35)
