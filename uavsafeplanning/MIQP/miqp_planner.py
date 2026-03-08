"""
MIQP Planner Wrapper for Fair Comparison with STO

This module provides a clean interface to MIQPTrajectoryPlanner that matches
the STOPlanner interface for easy comparison.

Key features:
- Fixed time allocation (predetermined)
- Hard constraints via binary variables
- Minimize jerk objective
- Bezier control points for dynamic/corridor constraints
"""

import numpy as np
import time as time_module
from MIQP_MINCO_solver import MIQPTrajectoryPlanner


class MIQPMincoPlanner:
    """
    Wrapper for MIQP trajectory planner with a clean interface for comparison.
    
    This class provides a simple interface similar to STOPlanner for fair
    comparison between gradient-based (STO) and integer programming (MIQP) methods.
    """
    
    def __init__(
        self,
        n_segments: int,
        A_list: list,
        b_list: list,
        time_allocation: np.ndarray,
        v_max: float,
        a_max: float,
        time_limit: float = 60.0,
        threads: int = 4,
        verbose: bool = True
    ):
        """
        Initialize MIQP planner.
        
        Parameters
        ----------
        n_segments : int
            Number of trajectory segments
        A_list : list
            List of corridor constraint matrices (H-representation)
        b_list : list
            List of corridor constraint vectors (H-representation)
        time_allocation : np.ndarray
            Fixed time per segment (MIQP requires fixed time)
        v_max : float
            Maximum velocity (m/s)
        a_max : float
            Maximum acceleration (m/s²)
        time_limit : float, optional
            Gurobi solver time limit (seconds)
        threads : int, optional
            Number of threads for Gurobi
        verbose : bool, optional
            Print optimization progress
        """
        self.n_segments = n_segments
        self.A_list = A_list
        self.b_list = b_list
        self.time_allocation = time_allocation
        self.v_max = v_max
        self.a_max = a_max
        self.time_limit = time_limit
        self.threads = threads
        self.verbose = verbose
        
        # Compute average dt (assuming uniform time allocation for MIQP)
        self.dt = np.mean(time_allocation)
        self.total_time = np.sum(time_allocation)
        
        # Initialize MIQP solver
        self.planner = MIQPTrajectoryPlanner(
            N=n_segments,
            dt=self.dt,
            A_list=A_list,
            b_list=b_list,
            v_max=v_max,
            a_max=a_max,
            verbose=verbose,
            time_limit=time_limit,
            threads=threads
        )
        
        # Results storage
        self.solved = False
        self.runtime = 0.0
        self.jerk_cost = 0.0
        self.optimality_gap = 0.0
        
        if verbose:
            print("=" * 60)
            print("MIQP TRAJECTORY OPTIMIZATION")
            print("=" * 60)
            print(f"Segments: {n_segments}")
            print(f"Time per segment: {self.dt:.3f} s (fixed)")
            print(f"Total time: {self.total_time:.3f} s")
            print(f"Solver: Gurobi (MIQP)")
            print(f"Time limit: {time_limit:.1f} s")
            print(f"Threads: {threads}")
            print("=" * 60)
    
    def solve(
        self,
        pos_init: np.ndarray,
        vel_init: np.ndarray,
        acc_init: np.ndarray,
        pos_final: np.ndarray,
        vel_final: np.ndarray,
        acc_final: np.ndarray
    ) -> bool:
        """
        Solve MIQP trajectory optimization problem.
        
        Parameters
        ----------
        pos_init : np.ndarray
            Initial position [x, y, z]
        vel_init : np.ndarray
            Initial velocity [vx, vy, vz]
        acc_init : np.ndarray
            Initial acceleration [ax, ay, az]
        pos_final : np.ndarray
            Final position [x, y, z]
        vel_final : np.ndarray
            Final velocity [vx, vy, vz]
        acc_final : np.ndarray
            Final acceleration [ax, ay, az]
            
        Returns
        -------
        bool
            True if optimization succeeded, False otherwise
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("MIQP OPTIMIZATION")
            print("=" * 60)
            print("\nSolving MIQP problem with Gurobi...")
        
        # Solve
        start_time = time_module.time()
        success = self.planner.solve(
            pos_init=pos_init,
            vel_init=vel_init,
            acc_init=acc_init,
            pos_final=pos_final,
            vel_final=vel_final,
            acc_final=acc_final
        )
        self.runtime = time_module.time() - start_time
        
        self.solved = success
        
        if success:
            # Extract optimization metrics
            if hasattr(self.planner, 'model') and self.planner.model is not None:
                self.jerk_cost = self.planner.model.ObjVal
                self.optimality_gap = self.planner.model.MIPGap
            
            if self.verbose:
                print("\n" + "=" * 60)
                print("✓ MIQP OPTIMIZATION COMPLETE!")
                print("=" * 60)
                print(f"Runtime: {self.runtime:.3f} seconds")
                print(f"Jerk cost: {self.jerk_cost:.6f}")
                print(f"Optimality gap: {self.optimality_gap:.6f}")
                print("=" * 60)
        else:
            if self.verbose:
                print("\n" + "=" * 60)
                print("✗ MIQP OPTIMIZATION FAILED!")
                print("=" * 60)
        
        return success
    
    def get_results(self, n_samples: int = 200) -> dict:
        """
        Extract optimization results for visualization and analysis.
        
        Parameters
        ----------
        n_samples : int, optional
            Number of samples for trajectory evaluation
            
        Returns
        -------
        dict
            Dictionary containing:
            - waypoints: Optimized waypoints (N-1, 3)
            - time_allocation: Time per segment (N,)
            - total_time: Total trajectory time
            - pos, vel, acc, jerk: Sampled trajectory (n_samples, 3)
            - vel_norm, acc_norm, jerk_norm: Norms (n_samples,)
            - t_eval: Time samples (n_samples,)
            - violations: Constraint violation statistics
            - runtime: Optimization runtime
            - jerk_cost: Jerk objective value
            - optimality_gap: MIP optimality gap
        """
        if not self.solved:
            raise RuntimeError("Cannot get results: MIQP optimization not solved successfully!")
        
        # Sample trajectory
        t_eval, pos, vel, acc = self.planner.get_trajectory(num_samples=n_samples)
        
        # Compute jerk numerically (finite differences)
        dt = np.diff(t_eval)
        jerk = np.zeros_like(acc)
        for i in range(len(dt)):
            if i < len(acc) - 1:
                jerk[i] = (acc[i+1] - acc[i]) / dt[i]
        jerk[-1] = jerk[-2]  # Copy last value
        
        # Ensure all arrays have same length
        actual_samples = min(pos.shape[0], vel.shape[0], acc.shape[0], jerk.shape[0], len(t_eval))
        pos = pos[:actual_samples]
        vel = vel[:actual_samples]
        acc = acc[:actual_samples]
        jerk = jerk[:actual_samples]
        t_eval = t_eval[:actual_samples]
        
        # Compute norms
        vel_norm = np.linalg.norm(vel, axis=1)
        acc_norm = np.linalg.norm(acc, axis=1)
        jerk_norm = np.linalg.norm(jerk, axis=1)
        
        # Check constraint violations
        vel_violations = np.maximum(0, vel_norm - self.v_max)
        acc_violations = np.maximum(0, acc_norm - self.a_max)
        
        max_vel_viol = vel_violations.max()
        max_acc_viol = acc_violations.max()
        
        # Check corridor violations
        # IMPORTANT: Check if position is inside ANY corridor, not just time-based corridor
        # MIQP allows segments to be in multiple corridors via binary variables
        corridor_viols = []
        n_corridors = len(self.A_list)
        
        for i in range(actual_samples):
            pos_sample = pos[i]
            
            # Check if position is inside ANY corridor
            inside_any_corridor = False
            min_violation = float('inf')
            
            for corridor_idx in range(n_corridors):
                A_corr = self.A_list[corridor_idx]
                b_corr = self.b_list[corridor_idx].flatten()
                
                # Check: A @ pos <= b (with small tolerance for numerical errors)
                distances = A_corr @ pos_sample - b_corr
                max_dist = distances.max()
                
                # Track minimum violation across all corridors
                if max_dist < min_violation:
                    min_violation = max_dist
                
                # If inside this corridor (with 1cm tolerance), no violation
                if max_dist <= 0.01:
                    inside_any_corridor = True
                    break
            
            # If not inside any corridor, record the violation distance
            if not inside_any_corridor:
                corridor_viols.append(min_violation)
        
        max_corr_viol = max(corridor_viols) if len(corridor_viols) > 0 else 0.0
        
        # Extract waypoints (internal knot points from polynomial coefficients)
        # For MIQP, we don't have explicit waypoints, so we sample at segment boundaries
        waypoints = []
        for i in range(1, self.n_segments):
            t_waypoint = np.sum(self.time_allocation[:i])
            idx = np.argmin(np.abs(t_eval - t_waypoint))
            waypoints.append(pos[idx])
        waypoints = np.array(waypoints) if waypoints else np.zeros((0, 3))
        
        return {
            'waypoints': waypoints,
            'time_allocation': self.time_allocation,
            'total_time': self.total_time,
            'pos': pos,
            'vel': vel,
            'acc': acc,
            'jerk': jerk,
            'vel_norm': vel_norm,
            'acc_norm': acc_norm,
            'jerk_norm': jerk_norm,
            't_eval': t_eval,
            'violations': {
                'vel': max_vel_viol,
                'acc': max_acc_viol,
                'corridor': max_corr_viol
            },
            'runtime': self.runtime,
            'jerk_cost': self.jerk_cost,
            'optimality_gap': self.optimality_gap
        }
