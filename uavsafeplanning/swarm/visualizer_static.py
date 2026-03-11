"""Matplotlib-based static visualizations for multi-UAV trajectories."""

from __future__ import annotations

from itertools import combinations
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .uav import UAV

# Distinct colours for up to 10 UAVs
_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _uav_color(idx: int) -> str:
    return _COLORS[idx % len(_COLORS)]


# -------------------------------------------------------------------
# 3-D trajectory plot
# -------------------------------------------------------------------

def plot_trajectories_3d(
    uavs: List[UAV],
    *,
    A_list: Optional[list] = None,
    b_list: Optional[list] = None,
    time_marker: Optional[float] = None,
    title: str = "Multi-UAV Trajectories",
    figsize: tuple = (12, 10),
    show: bool = True,
):
    """3-D plot of all UAV trajectories with optional corridor overlay.

    Parameters
    ----------
    uavs : list of UAV
    A_list, b_list : optional corridor constraints (from first UAV or shared).
    time_marker : float, optional
        If given, draw a sphere on each trajectory at this global time.
    title : str
    figsize : tuple
    show : bool
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Corridors (optional)
    if A_list is not None and b_list is not None:
        try:
            import pydecomp as pdc
            pdc.visualize_environment(A_list, b_list, ax=ax)
        except Exception:
            pass

    for idx, uav in enumerate(uavs):
        if uav.sto_results is None:
            continue
        color = _uav_color(idx)
        pos = uav.sto_results["pos"]
        ax.plot(
            pos[:, 0], pos[:, 1], pos[:, 2],
            color=color, linewidth=2, label=uav.uav_id,
        )
        ax.scatter(*uav.start, color=color, marker="o", s=100, edgecolors="black", zorder=5)
        ax.scatter(*uav.goal, color=color, marker="*", s=150, edgecolors="black", zorder=5)

        if time_marker is not None:
            p = uav.sample_position(time_marker)
            if p is not None:
                ax.scatter(*p, color=color, s=200, marker="D", edgecolors="white", linewidths=1.5, zorder=10)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    ax.view_init(elev=50, azim=-30)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


# -------------------------------------------------------------------
# Minimum inter-UAV distance over time
# -------------------------------------------------------------------

def plot_min_distances(
    uavs: List[UAV],
    dist_times: np.ndarray,
    pair_dists: np.ndarray,
    safety_margin: float = 0.0,
    *,
    title: str = "Inter-UAV Distance Over Time",
    figsize: tuple = (12, 5),
    show: bool = True,
):
    """Plot pairwise Euclidean distance over time.

    Parameters
    ----------
    uavs : list of UAV
    dist_times : ndarray of shape (T,)
    pair_dists : ndarray of shape (n_pairs, T)
    safety_margin : float
        Drawn as a dashed red line (sum of radii + margin).
    """
    fig, ax = plt.subplots(figsize=figsize)

    pair_idx = 0
    for i in range(len(uavs)):
        for j in range(i + 1, len(uavs)):
            label = f"{uavs[i].uav_id} <-> {uavs[j].uav_id}"
            threshold = uavs[i].radius + uavs[j].radius + safety_margin
            ax.plot(dist_times, pair_dists[pair_idx], linewidth=2, label=label)
            pair_idx += 1

    # Draw combined safety threshold (use max pair radius sum for a single line)
    all_thresholds = [
        uavs[i].radius + uavs[j].radius + safety_margin
        for i in range(len(uavs)) for j in range(i + 1, len(uavs))
    ]
    if all_thresholds:
        ax.axhline(
            max(all_thresholds), color="red", linestyle="--", linewidth=2,
            label=f"Safety threshold ({max(all_thresholds):.1f} m)",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


# -------------------------------------------------------------------
# Per-UAV velocity / acceleration profiles
# -------------------------------------------------------------------

def plot_velocity_profiles(
    uavs: List[UAV],
    v_max: float = 2.0,
    a_max: float = 1.0,
    *,
    title: str = "Dynamic Profiles",
    figsize: tuple = (14, 5),
    show: bool = True,
):
    """Stacked velocity and acceleration profiles for each UAV."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for idx, uav in enumerate(uavs):
        if uav.sto_results is None:
            continue
        color = _uav_color(idx)
        t_eval = uav.sto_results["t_eval"] + uav.time_offset
        vel_norm = uav.sto_results["vel_norm"]
        acc_norm = uav.sto_results["acc_norm"]

        axes[0].plot(t_eval, vel_norm, color=color, linewidth=1.5, label=uav.uav_id)
        axes[1].plot(t_eval, acc_norm, color=color, linewidth=1.5, label=uav.uav_id)

    axes[0].axhline(v_max, color="red", linestyle="--", linewidth=2, label=f"v_max={v_max}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].set_title("Velocity")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(a_max, color="red", linestyle="--", linewidth=2, label=f"a_max={a_max}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Acceleration (m/s²)")
    axes[1].set_title("Acceleration")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes


# -------------------------------------------------------------------
# Time-synced snapshot at a given t
# -------------------------------------------------------------------

def plot_snapshot_3d(
    uavs: List[UAV],
    t: float,
    *,
    trail_alpha: float = 0.3,
    title: str | None = None,
    figsize: tuple = (10, 9),
    show: bool = True,
):
    """3-D snapshot showing UAV positions at global time *t* with faded trails."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    for idx, uav in enumerate(uavs):
        if uav.sto_results is None:
            continue
        color = _uav_color(idx)
        pos = uav.sto_results["pos"]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=color, alpha=trail_alpha, linewidth=1)

        p = uav.sample_position(t)
        if p is not None:
            ax.scatter(*p, color=color, s=250, marker="o", edgecolors="black", linewidths=1.5, label=uav.uav_id, zorder=10)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title or f"Swarm Snapshot at t = {t:.2f} s")
    ax.legend()
    ax.view_init(elev=50, azim=-30)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


# -------------------------------------------------------------------
# Summary dashboard
# -------------------------------------------------------------------

def plot_swarm_dashboard(
    uavs: List[UAV],
    swarm_results: dict,
    v_max: float = 2.0,
    a_max: float = 1.0,
    safety_margin: float = 0.0,
    *,
    show: bool = True,
):
    """Convenience wrapper that calls the individual plot functions.

    Parameters
    ----------
    swarm_results : dict
        Output of ``SwarmPlanner.plan_all()``.
    """
    dist_times, pair_dists = swarm_results["min_distance_profile"]

    plot_trajectories_3d(uavs, title="Multi-UAV Trajectories (3D)", show=show)
    plot_min_distances(
        uavs, dist_times, pair_dists,
        safety_margin=safety_margin, show=show,
    )
    plot_velocity_profiles(uavs, v_max=v_max, a_max=a_max, show=show)
