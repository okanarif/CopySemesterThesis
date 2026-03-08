"""
Comparison Visualization Utilities
====================================

Provides side-by-side comparison tools for trajectory optimization methods
(STO vs MIQP).

Functions:
- compare_metrics: Summary metrics table
- compare_dynamic_profiles: Velocity/acceleration/jerk comparison
- compare_trajectories_3d: Side-by-side 3D trajectory plots
- compare_orthogonal_views: XY, XZ, YZ plane views (STO left, MIQP right)
- visualize_comparison: Comprehensive comparison (all-in-one)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import pydecomp as pdc


def compare_trajectories_3d(
    results_dict: dict,
    A_list: list,
    b_list: list,
    start: np.ndarray,
    goal: np.ndarray,
    figsize: tuple = None
):
    """
    Side-by-side 3D comparison of multiple trajectory optimization methods.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping method names to result dictionaries.
        Each result dict should contain:
        - 'pos': trajectory positions (N_samples, 3)
        - 'vel': trajectory velocities (N_samples, 3)
        - 'waypoints': waypoint positions (N_waypoints, 3)
    A_list : list
        Corridor A matrices
    b_list : list
        Corridor b vectors
    start : np.ndarray
        Start position (3,)
    goal : np.ndarray
        Goal position (3,)
    figsize : tuple, optional
        Figure size (width, height)
    """
    n_methods = len(results_dict)
    if figsize is None:
        figsize = (8 * n_methods, 8)
    
    fig = plt.figure(figsize=figsize)
    
    for idx, (method_name, results) in enumerate(results_dict.items(), start=1):
        ax = fig.add_subplot(1, n_methods, idx, projection='3d')
        
        pos = results['pos']
        vel = results['vel']
        waypoints = results['waypoints']
        
        # Compute speed for color mapping
        speed = np.linalg.norm(vel, axis=1)
        norm = plt.Normalize(vmin=speed.min(), vmax=speed.max())
        cmap = plt.cm.plasma
        points = pos.reshape(-1, 1, 3)
        
        # Safe corridors
        pdc.visualize_environment(A_list, b_list, ax=ax)
        
        # Trajectory (speed-colored)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(speed[:-1])
        lc.set_linewidth(3)
        ax.add_collection3d(lc)
        
        # Colorbar
        fig.colorbar(lc, ax=ax, label='Speed [m/s]', shrink=0.5)
        
        # Waypoints
        ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                  c='cyan', marker='o', s=50, edgecolors='black', linewidths=0.5,
                  label=f'Waypoints ({len(waypoints)})')
        
        # Start and goal
        ax.scatter(*start, c='lime', s=150, marker='o', 
                  edgecolors='black', linewidths=2, label='Start')
        ax.scatter(*goal, c='red', s=200, marker='*', 
                  edgecolors='black', linewidths=1, label='Goal')
        
        ax.view_init(elev=60, azim=-20)
        ax.legend(loc='upper left', fontsize=9)
        ax.set_title(f"{method_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        
        # Add stats text box (right side, text left-aligned inside)
        runtime = results.get('runtime', 0)
        total_time = results.get('total_time', 0)
        
        stats_text = f"Runtime:   {runtime:.2f}s\n"
        stats_text += f"Total time: {total_time:.2f}s"
        
        ax.text2D(0.98, 0.98, stats_text, transform=ax.transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def compare_orthogonal_views(
    results_dict: dict,
    A_list: list,
    b_list: list,
    start: np.ndarray,
    goal: np.ndarray,
    figsize: tuple = (14, 14)
):
    """
    Orthogonal view comparison: XY, XZ, YZ planes.
    Left column: STO, Right column: MIQP.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with keys e.g. 'STO', 'MIQP'
        Each value: dict with 'pos', 'vel', 'waypoints'
    A_list, b_list : list
        Corridor constraints
    start, goal : np.ndarray
        Start and goal positions (3,)
    figsize : tuple
        Figure size (width, height)
    """
    # Ensure exactly 2 methods, order: STO left, MIQP right
    items = list(results_dict.items())
    if len(items) != 2:
        items = items[:2]  # Take first two if more
    
    # Sort so STO is first (left), MIQP second (right)
    def sort_key(x):
        name = x[0].upper()
        if 'STO' in name:
            return 0
        if 'MIQP' in name:
            return 1
        return 0
    
    items = sorted(items, key=sort_key)
    
    views = [
        (90, -90, "XY (Top-down)"),
        (0, 90, "XZ (Side)"),
        (0, 0, "YZ (Side)")
    ]
    
    fig = plt.figure(figsize=figsize)
    
    for row_idx, (elev, azim, view_title) in enumerate(views):
        for col_idx, (method_name, results) in enumerate(items):
            subplot_idx = row_idx * 2 + col_idx + 1
            ax = fig.add_subplot(3, 2, subplot_idx, projection='3d')
            
            pos = results['pos']
            vel = results['vel']
            waypoints = results['waypoints']
            
            speed = np.linalg.norm(vel, axis=1)
            norm = plt.Normalize(vmin=speed.min(), vmax=speed.max())
            cmap = plt.cm.plasma
            points = pos.reshape(-1, 1, 3)
            
            # Corridors
            pdc.visualize_environment(A_list, b_list, ax=ax)
            
            # Trajectory
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = Line3DCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(speed[:-1])
            lc.set_linewidth(2)
            ax.add_collection3d(lc)
            
            # Waypoints
            ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                      c='cyan', marker='o', s=30, edgecolors='black', linewidths=0.3)
            
            # Start/goal
            ax.scatter(*start, c='lime', s=100, marker='o',
                      edgecolors='black', linewidths=1)
            ax.scatter(*goal, c='red', s=120, marker='*',
                      edgecolors='black', linewidths=0.8)
            
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{method_name} - {view_title}", fontsize=12)
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_zlabel('Z [m]')
    
    plt.suptitle("Orthogonal Views (STO | MIQP)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def compare_dynamic_profiles(
    results_dict: dict,
    v_max: float,
    a_max: float,
    j_max: float = None,
    figsize: tuple = (18, 12)
):
    """
    Compare velocity, acceleration, and jerk profiles across methods.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping method names to result dictionaries.
        Each result dict should contain:
        - 't_eval': time samples (N_samples,)
        - 'vel': velocities (N_samples, 3)
        - 'acc': accelerations (N_samples, 3)
        - 'jerk': jerks (N_samples, 3)
        Or precomputed norms:
        - 'vel_norm', 'acc_norm', 'jerk_norm'
    v_max : float
        Maximum velocity limit
    a_max : float
        Maximum acceleration limit
    j_max : float, optional
        Maximum jerk limit. If None (default), no limit line is drawn
        (jerk is unconstrained in both STO and MIQP).
    figsize : tuple
        Figure size (width, height)
    """
    n_methods = len(results_dict)
    
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    linestyles = ['-', '--', '-.', ':', '-']
    
    for idx, (method_name, results) in enumerate(results_dict.items()):
        # MIQP: solid purple line
        if 'MIQP' in method_name:
            color = 'purple'
            linestyle = '-'
        else:
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
        t_eval = results['t_eval']
        
        # Compute norms if not provided
        if 'vel_norm' in results:
            vel_norm = results['vel_norm']
        else:
            vel_norm = np.linalg.norm(results['vel'], axis=1)
        
        if 'acc_norm' in results:
            acc_norm = results['acc_norm']
        else:
            acc_norm = np.linalg.norm(results['acc'], axis=1)
        
        if 'jerk_norm' in results:
            jerk_norm = results['jerk_norm']
        else:
            jerk_norm = np.linalg.norm(results['jerk'], axis=1)
        
        # Velocity
        axes[0].plot(t_eval, vel_norm, color=color, linestyle=linestyle,
                    linewidth=2, label=method_name)
        
        # Acceleration
        axes[1].plot(t_eval, acc_norm, color=color, linestyle=linestyle,
                    linewidth=2, label=method_name)
        
        # Jerk
        axes[2].plot(t_eval, jerk_norm, color=color, linestyle=linestyle,
                    linewidth=2, label=method_name)
    
    # Green zone: extend to max total time across all methods
    t_min = min(r['t_eval'][0] for r in results_dict.values())
    t_max = max(r['t_eval'][-1] for r in results_dict.values())
    
    # Velocity plot
    axes[0].axhline(y=v_max, color='red', linestyle='--', linewidth=2, 
                   label=f'v_max = {v_max} m/s', alpha=0.7)
    axes[0].fill_between([t_min, t_max], 0, v_max, alpha=0.1, color='green')
    axes[0].set_ylabel('Velocity [m/s]', fontsize=12)
    axes[0].set_title('Velocity Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right', fontsize=10)
    
    # Acceleration plot
    axes[1].axhline(y=a_max, color='red', linestyle='--', linewidth=2,
                   label=f'a_max = {a_max} m/s²', alpha=0.7)
    axes[1].fill_between([t_min, t_max], 0, a_max, alpha=0.1, color='green')
    axes[1].set_ylabel('Acceleration [m/s²]', fontsize=12)
    axes[1].set_title('Acceleration Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=10)
    
    # Jerk plot (no constraint line — jerk is unconstrained)
    if j_max is not None:
        axes[2].axhline(y=j_max, color='red', linestyle='--', linewidth=2,
                       label=f'j_max = {j_max} m/s³', alpha=0.7)
        axes[2].fill_between([t_min, t_max], 0, j_max, alpha=0.1, color='green')
    axes[2].set_xlabel('Time [s]', fontsize=12)
    axes[2].set_ylabel('Jerk [m/s³]', fontsize=12)
    axes[2].set_title('Jerk Comparison', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def compare_metrics(results_dict: dict):
    """
    Display a comparison table of optimization metrics.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping method names to result dictionaries.
        Each result dict should contain:
        - 'runtime': optimization time (seconds)
        - 'jerk_cost': final jerk cost
        - 'total_time': trajectory duration (seconds)
        - 'violations': dict with 'vel', 'acc', 'jerk', 'corridor'
    """
    print("\n" + "="*80)
    print("TRAJECTORY OPTIMIZATION COMPARISON")
    print("="*80)
    
    # Header
    header = f"{'Method':<20} | {'Runtime (s)':<12} | {'Jerk Cost':<12} | {'Traj Time (s)':<14}"
    print(header)
    print("-"*80)
    
    # Data rows
    for method_name, results in results_dict.items():
        runtime = results.get('runtime', 0)
        jerk_cost = results.get('jerk_cost', 0)
        total_time = results.get('total_time', 0)
        
        row = f"{method_name:<20} | {runtime:>12.3f} | {jerk_cost:>12.6f} | {total_time:>14.2f}"
        print(row)
    
    print("="*80)
    
    # Violations table
    print("\nCONSTRAINT VIOLATIONS (Max)")
    print("="*80)
    header_viol = f"{'Method':<20} | {'Velocity':<12} | {'Acceleration':<14} | {'Jerk':<12} | {'Corridor':<12}"
    print(header_viol)
    print("-"*80)
    
    for method_name, results in results_dict.items():
        violations = results.get('violations', {})
        vel_viol = violations.get('vel', 0)
        acc_viol = violations.get('acc', 0)
        jerk_viol = violations.get('jerk', 0)
        corr_viol = violations.get('corridor', 0)
        
        row_viol = f"{method_name:<20} | {vel_viol:>12.6f} | {acc_viol:>14.6f} | {jerk_viol:>12.6f} | {corr_viol:>12.6f}"
        print(row_viol)
    
    print("="*80 + "\n")


def visualize_comparison(
    results_dict: dict,
    A_list: list,
    b_list: list,
    start: np.ndarray,
    goal: np.ndarray,
    v_max: float,
    a_max: float,
    j_max: float = None
):
    """
    Comprehensive comparison: metrics, dynamic profiles, 3D trajectories, orthogonal views.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping method names to result dictionaries
    A_list, b_list : list
        Corridor constraints
    start, goal : np.ndarray
        Start and goal positions (3,)
    v_max, a_max : float
        Dynamic limits (velocity and acceleration)
    j_max : float, optional
        Jerk limit. If None (default), no jerk limit line is drawn.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE TRAJECTORY COMPARISON")
    print("="*80 + "\n")
    
    # 1. Metrics table
    compare_metrics(results_dict)
    
    # 2. Dynamic profiles
    print("Generating dynamic profile comparison...")
    compare_dynamic_profiles(results_dict, v_max, a_max, j_max)
    
    # 3. 3D trajectories
    print("Generating 3D trajectory comparison...")
    compare_trajectories_3d(results_dict, A_list, b_list, start, goal)
    
    # 4. Orthogonal views (XY, XZ, YZ)
    print("Generating orthogonal views (XY, XZ, YZ)...")
    compare_orthogonal_views(results_dict, A_list, b_list, start, goal)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80 + "\n")


# ====================================================================================
# HELPER FUNCTIONS
# ====================================================================================

def compute_violations(pos, vel, acc, jerk, A_list, b_list, v_max, a_max, j_max):
    """
    Compute constraint violations for a trajectory.
    
    Parameters
    ----------
    pos : np.ndarray
        Positions (N, 3)
    vel : np.ndarray
        Velocities (N, 3)
    acc : np.ndarray
        Accelerations (N, 3)
    jerk : np.ndarray
        Jerks (N, 3)
    A_list, b_list : list
        Corridor constraints
    v_max, a_max, j_max : float
        Dynamic limits
    
    Returns
    -------
    dict
        Violation statistics
    """
    vel_norm = np.linalg.norm(vel, axis=1)
    acc_norm = np.linalg.norm(acc, axis=1)
    jerk_norm = np.linalg.norm(jerk, axis=1)
    
    vel_violation = np.maximum(vel_norm - v_max, 0).max()
    acc_violation = np.maximum(acc_norm - a_max, 0).max()
    jerk_violation = np.maximum(jerk_norm - j_max, 0).max()
    
    # Corridor violations (check all corridors, take minimum violation)
    corridor_violation = 0.0
    for i in range(len(pos)):
        min_violation = float('inf')
        for A, b in zip(A_list, b_list):
            violation = np.maximum(A @ pos[i] - b, 0).sum()
            min_violation = min(min_violation, violation)
        corridor_violation = max(corridor_violation, min_violation)
    
    return {
        'vel': vel_violation,
        'acc': acc_violation,
        'jerk': jerk_violation,
        'corridor': corridor_violation
    }
