"""
Visualization utilities for trajectory planning comparison.
Supports both STO and MIQP methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import pydecomp as pdc


def compute_snap_from_jerk(jerk: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """
    Compute snap (4th derivative) from jerk using finite differences.
    
    Parameters
    ----------
    jerk : np.ndarray
        Jerk trajectory, shape (N_samples, 3)
    t_eval : np.ndarray
        Time samples, shape (N_samples,)
    
    Returns
    -------
    snap : np.ndarray
        Snap trajectory, shape (N_samples, 3)
    """
    dt = np.diff(t_eval)
    snap = np.zeros_like(jerk)
    
    for i in range(len(dt)):
        if i < len(jerk) - 1:
            snap[i] = (jerk[i+1] - jerk[i]) / dt[i]
    
    # Copy last value
    snap[-1] = snap[-2]
    
    return snap


def visualize_trajectory_3d(
    pos: np.ndarray,
    vel: np.ndarray,
    A_list: list,
    b_list: list,
    waypoints: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    title: str = "Optimized Trajectory",
    method_name: str = "Method",
    show_orthogonal_views: bool = True
):
    """
    Visualize 3D trajectory with safe flight corridors.
    
    Parameters
    ----------
    pos : np.ndarray
        Trajectory positions, shape (N_samples, 3)
    vel : np.ndarray
        Trajectory velocities, shape (N_samples, 3)
    A_list : list
        List of A matrices for corridor constraints
    b_list : list
        List of b vectors for corridor constraints
    waypoints : np.ndarray
        Waypoint positions, shape (N_waypoints+2, 3) [start, wps, goal]
    start : np.ndarray
        Start position (3,)
    goal : np.ndarray
        Goal position (3,)
    title : str
        Plot title
    method_name : str
        Method name for labels (e.g., "STO", "MIQP")
    show_orthogonal_views : bool
        Whether to show XY, XZ, YZ views
    """
    
    # Compute speed for color mapping
    speed = np.linalg.norm(vel, axis=1)
    norm = plt.Normalize(vmin=speed.min(), vmax=speed.max())
    cmap = plt.cm.plasma
    points = pos.reshape(-1, 1, 3)
    
    # ============================================================
    # Main 3D view
    # ============================================================
    
    fig = plt.figure(figsize=(10, 10))
    ax3d = fig.add_subplot(111, projection='3d')
    
    # Safe flight corridors
    pdc.visualize_environment(A_list, b_list, ax=ax3d)
    
    # Optimized trajectory (speed-colored line)
    segments_main = np.concatenate([points[:-1], points[1:]], axis=1)
    lc_main = Line3DCollection(segments_main, cmap=cmap, norm=norm)
    lc_main.set_array(speed[:-1])
    lc_main.set_linewidth(3)
    ax3d.add_collection3d(lc_main)
    
    # Colorbar for speed
    fig.colorbar(lc_main, ax=ax3d, label='Speed [m/s]')
    
    # Waypoints (segment connection points)
    n_waypoints = len(waypoints)
    ax3d.scatter(
        waypoints[:, 0],
        waypoints[:, 1],
        waypoints[:, 2],
        c='cyan',
        marker='o',
        s=50,
        edgecolors='black',
        linewidths=0.5,
        label=f'Waypoints ({n_waypoints})'
    )
    
    # Start and goal markers
    ax3d.scatter(*start, c='lime', s=150, marker='o', 
                edgecolors='black', linewidths=2, label='Start')
    ax3d.scatter(*goal, c='red', s=200, marker='*', 
                edgecolors='black', linewidths=1, label='Goal')
    
    ax3d.view_init(elev=60, azim=-20)
    ax3d.legend()
    ax3d.set_title(f"{title} - {method_name}")
    plt.show()
    
    # ============================================================
    # Orthogonal views (optional)
    # ============================================================
    
    if show_orthogonal_views:
        fig = plt.figure(figsize=(18, 6))
        
        views = [
            (90, -90, "XY View (Top-down)"),
            (0,   90, "XZ View (Side, +Y)"),
            (0,    0, "YZ View (Side, +X)")
        ]
        
        for i, (elev, azim, view_title) in enumerate(views, start=1):
            ax = fig.add_subplot(1, 3, i, projection='3d')
            
            # Safe corridors
            pdc.visualize_environment(A_list, b_list, ax=ax)
            
            # Trajectory
            segments_view = np.concatenate([points[:-1], points[1:]], axis=1)
            lc_view = Line3DCollection(segments_view, cmap=cmap, norm=norm)
            lc_view.set_array(speed[:-1])
            lc_view.set_linewidth(3)
            ax.add_collection3d(lc_view)
            
            # Waypoints
            ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                      c='cyan', marker='o', s=30, 
                      edgecolors='black', linewidths=0.3)
            
            # Start/goal
            ax.scatter(*start, c='lime', s=100, marker='o', 
                      edgecolors='black', linewidths=1)
            ax.scatter(*goal, c='red', s=120, marker='*', 
                      edgecolors='black', linewidths=0.8)
            
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(view_title)
            
            if i == 1:
                ax.legend(['Trajectory', 'Waypoints', 'Start', 'Goal'])
        
        plt.suptitle(f"{title} - {method_name} (Orthogonal Views)", fontsize=14)
        plt.tight_layout()
        plt.show()


def visualize_optimization_metrics(
    cost_history: dict,
    t_eval: np.ndarray,
    vel_norm: np.ndarray,
    acc_norm: np.ndarray,
    jerk_norm: np.ndarray,
    v_max: float,
    a_max: float,
    method_name: str = "Method",
    snap_norm: np.ndarray = None,
    s_max: float = None,
    weight_history: list = None,
):
    """
    Visualize optimization convergence and dynamic constraint profiles.
    Jerk is unconstrained in STO (only minimized as objective), so no limit line is shown.
    
    Parameters
    ----------
    cost_history : dict
        Dictionary with keys: 'total', 'jerk', 'time', 'vel', 'acc', 'corridor'
        Each value is a list of costs over iterations
    t_eval : np.ndarray
        Time samples for trajectory evaluation
    vel_norm : np.ndarray
        Velocity magnitudes at t_eval
    acc_norm : np.ndarray
        Acceleration magnitudes at t_eval
    jerk_norm : np.ndarray
        Jerk magnitudes at t_eval
    v_max : float
        Maximum velocity limit
    a_max : float
        Maximum acceleration limit
    method_name : str
        Method name for titles (e.g., "STO", "MIQP")
    snap_norm : np.ndarray, optional
        Snap magnitudes at t_eval
    s_max : float, optional
        Maximum snap limit
    weight_history : list of dict, optional
        Phase weight snapshots from STOPlanner.get_results()['weight_history'].
        When provided, vertical dashed lines are drawn at each phase boundary on
        the convergence plots.
    """
    
    # Determine number of rows based on whether snap is provided
    has_snap = snap_norm is not None
    n_rows = 3 if has_snap else 2
    fig = plt.figure(figsize=(15, 5 * n_rows))
    
    # ============================================================
    # Row 1: Optimization convergence
    # ============================================================
    
    # Helper: draw vertical phase-boundary lines on an axes
    def _draw_phase_lines(ax):
        if not weight_history or len(weight_history) <= 1:
            return
        # Phase boundaries are at iter_start of phases 2, 3, …
        for wh in weight_history[1:]:
            ax.axvline(x=wh['iter_start'] + 1, color='gray', linestyle='--',
                       linewidth=1.2, alpha=0.7,
                       label=f"Phase {wh['phase']} (λ_c={wh['lambda_corridor']:.0f})")

    # 1) Total cost history
    ax1 = plt.subplot(2, 3, 1)
    iterations = range(1, len(cost_history['total']) + 1)
    ax1.plot(iterations, cost_history['total'], 'b-', linewidth=2, label='Total Cost')
    _draw_phase_lines(ax1)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Total Cost')
    ax1.set_title(f'{method_name}: Total Cost History')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7)
    
    # 2) Primary cost components (jerk, time)
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(iterations, cost_history['jerk'], label='Jerk', linewidth=2)
    if 'time' in cost_history and len(cost_history['time']) > 0:
        ax2.plot(iterations, cost_history['time'], label='Time', linewidth=2)
    _draw_phase_lines(ax2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cost')
    ax2.set_title(f'{method_name}: Primary Costs')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7)
    
    # 3) Constraint penalty terms
    ax3 = plt.subplot(2, 3, 3)
    if 'vel' in cost_history:
        ax3.plot(iterations, cost_history['vel'], label='Velocity', linewidth=2)
    if 'acc' in cost_history:
        ax3.plot(iterations, cost_history['acc'], label='Acceleration', linewidth=2)
    if 'corridor' in cost_history:
        ax3.plot(iterations, cost_history['corridor'], label='Corridor', linewidth=2)
    _draw_phase_lines(ax3)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Penalty')
    ax3.set_title(f'{method_name}: Constraint Penalties')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=7)
    ax3.set_yscale('log')
    
    # ============================================================
    # Row 2: Dynamic constraint profiles
    # ============================================================
    
    # 4) Velocity profile
    ax4 = plt.subplot(n_rows, 3, 4)
    ax4.plot(t_eval, vel_norm, 'b-', linewidth=2, label='Velocity')
    ax4.axhline(y=v_max, color='r', linestyle='--', linewidth=2, 
               label=f'v_max = {v_max} m/s')
    ax4.fill_between(t_eval, 0, v_max, alpha=0.2, color='green')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.set_title(f'{method_name}: Velocity Profile')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5) Acceleration profile
    ax5 = plt.subplot(n_rows, 3, 5)
    ax5.plot(t_eval, acc_norm, 'g-', linewidth=2, label='Acceleration')
    ax5.axhline(y=a_max, color='r', linestyle='--', linewidth=2, 
               label=f'a_max = {a_max} m/s²')
    ax5.fill_between(t_eval, 0, a_max, alpha=0.2, color='green')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Acceleration (m/s²)')
    ax5.set_title(f'{method_name}: Acceleration Profile')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6) Jerk profile (unconstrained — no limit line)
    ax6 = plt.subplot(n_rows, 3, 6)
    ax6.plot(t_eval, jerk_norm, 'orange', linewidth=2, label='Jerk')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Jerk (m/s³)')
    ax6.set_title(f'{method_name}: Jerk Profile (unconstrained)')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # ============================================================
    # Row 3: Snap profile (if provided)
    # ============================================================
    
    if has_snap:
        # 7) Snap profile (centered in row 3)
        ax7 = plt.subplot(n_rows, 3, 8)  # Middle position of row 3
        ax7.plot(t_eval, snap_norm, 'purple', linewidth=2, label='Snap')
        
        if s_max is not None:
            ax7.axhline(y=s_max, color='r', linestyle='--', linewidth=2, 
                       label=f's_max = {s_max} m/s⁴')
            ax7.fill_between(t_eval, 0, s_max, alpha=0.2, color='green')
        
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Snap (m/s⁴)')
        ax7.set_title(f'{method_name}: Snap Profile (4th derivative)')
        ax7.grid(True, alpha=0.3)
        ax7.legend()
        
        # Add statistics
        max_snap = np.max(snap_norm)
        avg_snap = np.mean(snap_norm)
        ax7.text(0.02, 0.98, f'Max: {max_snap:.2f} m/s⁴\nAvg: {avg_snap:.2f} m/s⁴',
                transform=ax7.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def visualize_comparison(
    results_dict: dict,
    A_list: list,
    b_list: list,
    start: np.ndarray,
    goal: np.ndarray
):
    """
    Side-by-side comparison of multiple methods.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping method names to result dictionaries.
        Each result dict should contain:
        - 'pos': trajectory positions (N_samples, 3)
        - 'vel': trajectory velocities (N_samples, 3)
        - 'waypoints': waypoint positions (N_waypoints, 3)
        - 'runtime': computation time in seconds
        - 'jerk_cost': final jerk cost
        - 'violations': dict with 'vel', 'acc', 'jerk', 'corridor'
    A_list : list
        Corridor A matrices
    b_list : list
        Corridor b vectors
    start : np.ndarray
        Start position
    goal : np.ndarray
        Goal position
    """
    
    n_methods = len(results_dict)
    fig = plt.figure(figsize=(8 * n_methods, 8))
    
    for idx, (method_name, results) in enumerate(results_dict.items(), start=1):
        ax = fig.add_subplot(1, n_methods, idx, projection='3d')
        
        pos = results['pos']
        vel = results['vel']
        waypoints = results['waypoints']
        
        # Compute speed
        speed = np.linalg.norm(vel, axis=1)
        norm = plt.Normalize(vmin=speed.min(), vmax=speed.max())
        cmap = plt.cm.plasma
        points = pos.reshape(-1, 1, 3)
        
        # Safe corridors
        pdc.visualize_environment(A_list, b_list, ax=ax)
        
        # Trajectory
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(speed[:-1])
        lc.set_linewidth(3)
        ax.add_collection3d(lc)
        
        # Waypoints
        ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                  c='cyan', marker='o', s=50, edgecolors='black', linewidths=0.5)
        
        # Start/goal
        ax.scatter(*start, c='lime', s=150, marker='o', 
                  edgecolors='black', linewidths=2, label='Start')
        ax.scatter(*goal, c='red', s=200, marker='*', 
                  edgecolors='black', linewidths=1, label='Goal')
        
        ax.view_init(elev=60, azim=-20)
        
        # Add metrics to title
        runtime = results.get('runtime', 0)
        jerk = results.get('jerk_cost', 0)
        title = f"{method_name}\nTime: {runtime:.2f}s | Jerk: {jerk:.6f}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        if idx == 1:
            ax.legend()
    
    plt.suptitle("Method Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Method':<15} | {'Runtime (s)':<12} | {'Jerk Cost':<12} | {'Max Violations':<30}")
    print("-"*80)
    
    for method_name, results in results_dict.items():
        runtime = results.get('runtime', 0)
        jerk = results.get('jerk_cost', 0)
        viols = results.get('violations', {})
        
        vel_viol = viols.get('vel', 0)
        acc_viol = viols.get('acc', 0)
        corr_viol = viols.get('corridor', 0)
        
        viol_str = f"v:{vel_viol:.3f} a:{acc_viol:.3f} c:{corr_viol:.3f}"
        print(f"{method_name:<15} | {runtime:>12.3f} | {jerk:>12.6f} | {viol_str:<30}")
    
    print("="*80 + "\n")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
"""
Example usage with snap visualization:

# After getting trajectory results from MIQP or STO:
results = planner.get_results(n_samples=200)

pos = results['pos']
vel = results['vel']
acc = results['acc']
jerk = results['jerk']
t_eval = results['t_eval']

vel_norm = np.linalg.norm(vel, axis=1)
acc_norm = np.linalg.norm(acc, axis=1)
jerk_norm = np.linalg.norm(jerk, axis=1)

# Compute snap from jerk
snap = compute_snap_from_jerk(jerk, t_eval)
snap_norm = np.linalg.norm(snap, axis=1)

# Visualize with snap
visualize_optimization_metrics(
    cost_history=cost_history,
    t_eval=t_eval,
    vel_norm=vel_norm,
    acc_norm=acc_norm,
    jerk_norm=jerk_norm,
    v_max=5.0,
    a_max=3.0,
    method_name="STO",
    snap_norm=snap_norm,  # Optional: add snap profile
    s_max=50.0            # Optional: snap limit
)
"""
