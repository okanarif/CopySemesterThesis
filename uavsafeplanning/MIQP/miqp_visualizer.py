"""
Visualization utilities for MIQP trajectory planning.
Optimized for single-shot optimization (no iterative convergence plots).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import pydecomp as pdc


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
    t_eval: np.ndarray,
    vel_norm: np.ndarray,
    acc_norm: np.ndarray,
    jerk_norm: np.ndarray,
    v_max: float,
    a_max: float,
    method_name: str = "Method"
):
    """
    Visualize dynamic constraint profiles for MIQP.
    
    NOTE: This version is optimized for MIQP (single-shot optimization).
    No cost history plots since MIQP doesn't have iterative convergence.
    Shows only velocity, acceleration, and jerk profiles in a single row.
    Jerk is unconstrained in MIQP, so no limit line is shown for jerk.
    
    Parameters
    ----------
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
        Method name for titles (e.g., "MIQP")
    """
    
    # Single row with 3 plots (matching STO format)
    fig = plt.figure(figsize=(18, 5))
    
    # ============================================================
    # Dynamic Constraint Profiles (1 row × 3 columns)
    # ============================================================
    
    # 1) Velocity profile
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(t_eval, vel_norm, 'b-', linewidth=2, label='Velocity')
    ax1.axhline(y=v_max, color='r', linestyle='--', linewidth=2, 
               label=f'v_max = {v_max} m/s')
    ax1.fill_between(t_eval, 0, v_max, alpha=0.2, color='green')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title(f'{method_name}: Velocity Profile')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2) Acceleration profile
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(t_eval, acc_norm, 'g-', linewidth=2, label='Acceleration')
    ax2.axhline(y=a_max, color='r', linestyle='--', linewidth=2, 
               label=f'a_max = {a_max} m/s²')
    ax2.fill_between(t_eval, 0, a_max, alpha=0.2, color='green')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Acceleration (m/s²)')
    ax2.set_title(f'{method_name}: Acceleration Profile')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3) Jerk profile (unconstrained — no limit line)
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(t_eval, jerk_norm, 'orange', linewidth=2, label='Jerk')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Jerk (m/s³)')
    ax3.set_title(f'{method_name}: Jerk Profile (unconstrained)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
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
        jerk_viol = viols.get('jerk', 0)
        corr_viol = viols.get('corridor', 0)
        
        viol_str = f"v:{vel_viol:.3f} a:{acc_viol:.3f} j:{jerk_viol:.3f} c:{corr_viol:.3f}"
        print(f"{method_name:<15} | {runtime:>12.3f} | {jerk:>12.6f} | {viol_str:<30}")
    
    print("="*80 + "\n")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
"""
Example usage for MIQP visualization:

# After getting trajectory results from MIQP:
results = planner.get_results(n_samples=200)

# Visualize dynamic constraint profiles
visualize_optimization_metrics(
    t_eval=results['t_eval'],
    vel_norm=results['vel_norm'],
    acc_norm=results['acc_norm'],
    jerk_norm=results['jerk_norm'],
    v_max=5.0,
    a_max=3.0,
    method_name="MIQP"
)

# Visualize 3D trajectory
visualize_trajectory_3d(
    pos=results['pos'],
    vel=results['vel'],
    A_list=A_list,
    b_list=b_list,
    waypoints=results['waypoints'],
    start=pos_init,
    goal=pos_final,
    title="MIQP Trajectory",
    method_name="MIQP",
    show_orthogonal_views=True
)
"""
