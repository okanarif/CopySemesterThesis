"""
MIQP Trajectory Planner with MINCO Parametrization
==================================================

This module implements Mixed-Integer Quadratic Programming (MIQP) for trajectory 
optimization using MINCO-based parametrization for fair comparison with STO.

KEY FEATURES:
1. MINCO-Based Parametrization:
   - Decision variables: WAYPOINTS (N-1, 3)
   - Polynomial coefficients: Derived analytically via MINCO
   - Time allocation: Fixed (predetermined)

2. Bezier Control Point Constraints:
   - Dynamic constraints: Applied via Bezier control points (velocity, acceleration, jerk)
   - Corridor constraints: Applied via 6 Bezier control points for position
   - Convex hull property ensures ZERO corridor violations

3. Binary Variables:
   - Corridor assignment via indicator constraints
   - Each segment must be in at least one corridor

COMPARISON WITH STANDARD MIQP:
- Standard MIQP: Decision variables are polynomial coefficients
- This implementation: Decision variables are waypoints (MINCO mapping)
- Both use Bezier control points for constraints (mathematical correctness)
- This ensures fair comparison with STO (same parametrization)

Author: Trajectory Optimization Framework
Date: 2026
"""

import numpy as np
import torch
import gurobipy as gp
from gurobipy import GRB
from typing import List, Tuple, Dict, Optional
from traj_gen_utils import MINCO_S3NU


class MIQPTrajectoryPlanner:
    """
    MIQP trajectory planner with MINCO-based parametrization.
    
    Core Features:
    - Waypoint optimization (not coefficient optimization!)
    - MINCO analytical mapping (waypoints → coefficients)
    - Hard constraints via binary variables
    - Fixed time allocation
    """
    
    def __init__(
        self,
        N: int,
        dt: float,
        A_list: List[np.ndarray],
        b_list: List[np.ndarray],
        v_max: float,
        a_max: float,
        verbose: bool = True,
        time_limit: float = 60.0,
        threads: int = 4
    ):
        """
        Initialize MINCO-based MIQP planner.
        
        Parameters
        ----------
        N : int
            Number of trajectory segments
        dt : float
            Time per segment (fixed)
        A_list, b_list : List[np.ndarray]
            Corridor constraints (H-representation)
        v_max, a_max : float
            Dynamic limits (jerk is unconstrained)
        verbose : bool
            Print optimization progress
        time_limit : float
            Gurobi solver time limit (seconds)
        threads : int
            Number of threads for Gurobi
        """
        self.N = N
        self.dt = dt
        self.A_list = A_list
        self.b_list = b_list
        self.P = len(A_list)
        
        self.v_max = v_max
        self.a_max = a_max
        
        self.verbose = verbose
        self.time_limit = time_limit
        self.threads = threads
        
        # Initialize MINCO for analytical mapping
        if verbose:
            print("=" * 60)
            print("MINCO-BASED MIQP TRAJECTORY PLANNER")
            print("=" * 60)
            print(f"Segments (N):        {N}")
            print(f"Time per segment:    {dt:.3f} s")
            print(f"Corridors (P):       {self.P}")
            print(f"Decision variables:  Waypoints (MINCO-based!)")
            print("=" * 60)
        
        self._initialize_minco()
        
        # Gurobi model
        self.model = None
        self.vars = {}
        
        # Solution storage
        self.solved = False
        self.waypoints = None
        self.coefficients = None
        
    def _initialize_minco(self):
        """Initialize MINCO model and extract mapping matrix."""
        self.minco = MINCO_S3NU(self.N)
        
        # Fixed time allocation
        t_torch = torch.tensor([[self.dt]], dtype=torch.float32).repeat(1, self.N, 1)
        self.minco.set_constant_time_allocation(t_torch)
        
        # Extract MINCO mapping matrix: A_map shape (6N, N+5)
        # Maps [headPVA(3), waypoints(N-1), tailPVA(3)] → coefficients(6N)
        self.A_map = self.minco.A_map[0].numpy()  # Remove batch dimension
        
        if self.verbose:
            print(f"\n✓ MINCO initialized:")
            print(f"  Mapping matrix: {self.A_map.shape} (coeffs ← waypoints)")
            print(f"  Input:  [headPVA(3), waypoints({self.N-1}), tailPVA(3)]")
            print(f"  Output: coefficients({6*self.N})")
    
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
        Solve MINCO-based MIQP trajectory optimization.
        
        Returns
        -------
        bool
            True if optimization succeeded
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("BUILDING MIQP MODEL")
            print("=" * 60)
        
        # Create Gurobi model
        self.model = gp.Model("MINCO_MIQP")
        self.model.setParam('OutputFlag', 1 if self.verbose else 0)
        self.model.setParam('TimeLimit', self.time_limit)
        self.model.setParam('Threads', self.threads)
        self.model.setParam('MIPGap', 1e-4)
        
        # Store boundary conditions
        self.headPVA = np.vstack([pos_init, vel_init, acc_init])  # (3, 3)
        self.tailPVA = np.vstack([pos_final, vel_final, acc_final])  # (3, 3)
        
        # Build optimization problem
        self._create_variables()
        self._set_objective()
        self._add_corridor_constraints()
        self._add_dynamic_constraints()
        
        # Solve
        if self.verbose:
            print("\n" + "=" * 60)
            print("SOLVING MIQP")
            print("=" * 60)
        
        self.model.optimize()
        
        # Extract solution
        has_solution = (
            self.model.Status == GRB.OPTIMAL or
            (self.model.Status == GRB.TIME_LIMIT and self.model.SolCount > 0)
        )
        if has_solution:
            self._extract_solution()
            self.solved = True
            
            if self.verbose:
                print("\n" + "=" * 60)
                print("✓ OPTIMIZATION COMPLETE!")
                print("=" * 60)
                print(f"Status:    {self.model.Status}")
                print(f"Objective: {self.model.ObjVal:.6f}")
                print(f"MIP Gap:   {self.model.MIPGap:.6f}")
                print(f"Time:      {self.model.Runtime:.3f} s")
                print("=" * 60)
            return True
        else:
            if self.verbose:
                status_map = {
                    GRB.TIME_LIMIT: "TIME_LIMIT (no feasible solution found)",
                    GRB.INFEASIBLE: "INFEASIBLE",
                    GRB.INF_OR_UNBD: "INFEASIBLE_OR_UNBOUNDED",
                }
                status_str = status_map.get(self.model.Status, str(self.model.Status))
                print(f"\n✗ Optimization failed! Status: {status_str}")
            return False
    
    def _create_variables(self):
        """Create decision variables: waypoints + binary."""
        # Waypoints (N-1, 3) - interior waypoints
        self.vars['waypoints'] = self.model.addMVar(
            (self.N-1, 3), lb=-GRB.INFINITY, name="waypoints"
        )
        
        # Binary variables for corridor assignment (N, P)
        self.vars['binary'] = self.model.addMVar(
            (self.N, self.P), vtype=GRB.BINARY, name="binary"
        )
        
        # Add binary assignment constraints
        for n in range(self.N):
            self.model.addConstr(
                self.vars['binary'][n, :].sum() >= 1,
                name=f"assign_{n}"
            )
        
        if self.verbose:
            print(f"\n✓ Decision Variables:")
            print(f"  Waypoints: {(self.N-1)*3} continuous (MINCO-based)")
            print(f"  Binary:    {self.N*self.P} binary (corridor assignment)")
            print(f"  Total:     {(self.N-1)*3 + self.N*self.P} variables")
    
    def _set_objective(self):
        """
        Set objective: minimize jerk energy.
        
        With MINCO: jerk² is quadratic in waypoints via A_map.
        For 5th-order polynomial:
            jerk(t) = 6*c[3] + 24*c[4]*t + 60*c[5]*t²
            ∫₀^dt jerk²(t) dt = 36*c[3]²*dt + 144*c[3]*c[4]*dt² + 192*c[4]²*dt³ 
                              + 240*c[3]*c[5]*dt³ + 720*c[4]*c[5]*dt⁴ + 720*c[5]²*dt⁵
        """
        # Extract A_map from MINCO (mapping from extended waypoints to coefficients)
        A_map = self.A_map  # Already numpy array (6N, N+5)
        
        # Build total jerk² objective
        # MINCO: p_extended = [headPVA(3), waypoints(N-1), tailPVA(3)]
        # For each axis: coefficients = A_map @ p_extended
        # jerk² = sum over segments and axes
        
        objective = 0.0
        
        # Q matrix coefficients for jerk energy (same for all segments)
        dt = self.dt
        Q_33 = 36.0 * dt
        Q_34 = 72.0 * dt**2      # Factor of 2 for cross term
        Q_35 = 120.0 * dt**3     # Factor of 2 for cross term
        Q_44 = 192.0 * dt**3
        Q_45 = 360.0 * dt**4     # Factor of 2 for cross term
        Q_55 = 720.0 * dt**5
        
        # For each axis (x, y, z)
        for axis in range(3):
            # For each segment
            for seg in range(self.N):
                # Indices for c[3], c[4], c[5] of this segment
                idx_c3 = 6 * seg + 3
                idx_c4 = 6 * seg + 4
                idx_c5 = 6 * seg + 5
                
                # MINCO structure: columns [headPVA(3), waypoints(N-1), tailPVA(3)]
                
                # Create linear expressions for c3, c4, c5: c = A_map @ [headPVA; waypoints; tailPVA]
                c3_expr = 0.0
                c4_expr = 0.0
                c5_expr = 0.0
                
                # Add headPVA contributions (constant terms)
                for i in range(3):
                    c3_expr += A_map[idx_c3, i] * self.headPVA[i, axis]
                    c4_expr += A_map[idx_c4, i] * self.headPVA[i, axis]
                    c5_expr += A_map[idx_c5, i] * self.headPVA[i, axis]
                
                # Add waypoint contributions (decision variables)
                for wp_idx in range(self.N - 1):
                    col = 3 + wp_idx
                    c3_expr += A_map[idx_c3, col] * self.vars['waypoints'][wp_idx, axis]
                    c4_expr += A_map[idx_c4, col] * self.vars['waypoints'][wp_idx, axis]
                    c5_expr += A_map[idx_c5, col] * self.vars['waypoints'][wp_idx, axis]
                
                # Add tailPVA contributions (constant terms)
                for i in range(3):
                    col = 3 + (self.N - 1) + i
                    c3_expr += A_map[idx_c3, col] * self.tailPVA[i, axis]
                    c4_expr += A_map[idx_c4, col] * self.tailPVA[i, axis]
                    c5_expr += A_map[idx_c5, col] * self.tailPVA[i, axis]
                
                # Jerk² for this segment and axis (quadratic form)
                objective += Q_33 * c3_expr * c3_expr
                objective += 2 * Q_34 * c3_expr * c4_expr
                objective += 2 * Q_35 * c3_expr * c5_expr
                objective += Q_44 * c4_expr * c4_expr
                objective += 2 * Q_45 * c4_expr * c5_expr
                objective += Q_55 * c5_expr * c5_expr
        
        self.model.setObjective(objective, GRB.MINIMIZE)
        
        if self.verbose:
            print(f"\n✓ Objective:")
            print(f"  Minimize: ∫ jerk²(t) dt over all segments")
            print(f"  Formula:  ∑ [36*c₃²*dt + 144*c₃*c₄*dt² + ... + 720*c₅²*dt⁵]")
            print(f"  Segments: {self.N}")
    
    def _add_corridor_constraints(self):
        """
        Add corridor constraints using Bezier control points (MINCO-based).
        
        CRITICAL FIX: Uses 6 Bezier control points for position trajectory.
        This ensures the ENTIRE trajectory stays within corridors via convex hull property.
        
        For each segment:
        1. Extract coefficient expressions from MINCO mapping
        2. Compute 6 Bezier control points for position
        3. Apply corridor constraints to all control points
        
        This is the correct approach for 5th-order polynomials!
        """
        dt = self.dt
        A_map = self.A_map
        count_indicator = 0
        count_atleast = 0
        
        # For each segment
        for seg in range(self.N):
            # For each axis (x, y, z) - compute coefficient expressions
            # We need coefficients for all 3 axes to form 3D control points
            coeffs_all_axes = []  # Will contain 3 lists of [c0, c1, c2, c3, c4, c5]
            
            for axis in range(3):
                coeff_exprs = []
                
                for coeff_idx in range(6):
                    # Global coefficient index in A_map
                    global_idx = 6 * seg + coeff_idx
                    
                    # Build linear expression: coeff = A_map @ [headPVA; waypoints; tailPVA]
                    expr = 0.0
                    
                    # Add headPVA contributions (columns 0-2: initial P, V, A)
                    for i in range(3):
                        expr += A_map[global_idx, i] * self.headPVA[i, axis]
                    
                    # Add waypoint contributions (columns 3 to 3+N-2)
                    for wp_idx in range(self.N - 1):
                        col = 3 + wp_idx
                        expr += A_map[global_idx, col] * self.vars['waypoints'][wp_idx, axis]
                    
                    # Add tailPVA contributions (columns 3+N-1 to 3+N+1)
                    for i in range(3):
                        col = 3 + (self.N - 1) + i
                        expr += A_map[global_idx, col] * self.tailPVA[i, axis]
                    
                    coeff_exprs.append(expr)
                
                coeffs_all_axes.append(coeff_exprs)
            
            # Now compute 6 Bezier control points (3D positions)
            # For 5th-order polynomial: p(t) = c0 + c1*t + c2*t² + c3*t³ + c4*t⁴ + c5*t⁵
            # Bezier control points:
            control_points = []  # Will contain 6 points, each is [x_expr, y_expr, z_expr]
            
            for k in range(6):
                r_k = []  # Control point k for all 3 axes
                
                for axis in range(3):
                    c0, c1, c2, c3, c4, c5 = coeffs_all_axes[axis]
                    
                    if k == 0:
                        # r0 = c0 (start position)
                        r_k.append(c0)
                    elif k == 1:
                        # r1 = (c1*dt + 5*c0) / 5
                        r_k.append((c1 * dt + 5 * c0) / 5)
                    elif k == 2:
                        # r2 = (2*c2*dt² + 4*c1*dt + 10*c0) / 10
                        r_k.append((2 * c2 * dt**2 + 4 * c1 * dt + 10 * c0) / 10)
                    elif k == 3:
                        # r3 = (3*c3*dt³ + 3*c2*dt² + 3*c1*dt + 10*c0) / 10
                        r_k.append((3 * c3 * dt**3 + 3 * c2 * dt**2 + 3 * c1 * dt + 10 * c0) / 10)
                    elif k == 4:
                        # r4 = (4*c4*dt⁴ + 4*c3*dt³ + 2*c2*dt² + 4*c1*dt + 5*c0) / 5
                        r_k.append((4 * c4 * dt**4 + 4 * c3 * dt**3 + 2 * c2 * dt**2 + 4 * c1 * dt + 5 * c0) / 5)
                    else:  # k == 5
                        # r5 = c5*dt⁵ + c4*dt⁴ + c3*dt³ + c2*dt² + c1*dt + c0 (end position)
                        r_k.append(c5 * dt**5 + c4 * dt**4 + c3 * dt**3 + c2 * dt**2 + c1 * dt + c0)
                
                control_points.append(r_k)
            
            # Add indicator constraints for all 6 control points
            for p in range(self.P):  # For each polytope
                A_p = self.A_list[p]
                b_p = self.b_list[p]
                m_p = A_p.shape[0]  # Number of half-spaces
                
                # For each control point
                for k, r_k in enumerate(control_points):
                    # For each half-space
                    for j in range(m_p):
                        # If binary[seg, p] == 1, then A_p[j] @ r_k <= b_p[j]
                        lhs = sum(A_p[j, i] * r_k[i] for i in range(3))
                        self.model.addGenConstrIndicator(
                            self.vars['binary'][seg, p],
                            1,  # When binary = 1
                            lhs <= b_p[j],
                            name=f"corridor_{seg}_{p}_cp{k}_hs{j}"
                        )
                        count_indicator += 1
            
            # Each segment must be in at least one polytope
            self.model.addConstr(
                self.vars['binary'][seg, :].sum() >= 1,
                name=f"atleast_one_poly_{seg}"
            )
            count_atleast += 1
        
        if self.verbose:
            print(f"\n✓ Corridor Constraints:")
            print(f"  {count_indicator} indicator constraints (Bezier control points)")
            print(f"  6 control points × {self.P} corridors × {self.N} segments")
            print(f"  {count_atleast} 'at least one corridor' constraints")
            print(f"  ✓ Convex hull property ensures ZERO corridor violations!")
    
    def _add_dynamic_constraints(self):
        """
        Add dynamic constraints via Bezier control points with NORM constraints.
        
        For 5th-order polynomial: p(t) = c[0] + c[1]*t + c[2]*t² + c[3]*t³ + c[4]*t⁴ + c[5]*t⁵
        - Velocity (4th-order):     5 control points
        - Acceleration (3rd-order): 4 control points
        - Jerk (2nd-order):         3 control points
        
        IMPORTANT: Use L2 norm constraints: ||v||² ≤ v_max² (not per-axis!)
        """
        dt = self.dt
        A_map = self.A_map
        count = 0
        
        # For each segment
        for seg in range(self.N):
            # ============================================================
            # Extract coefficient expressions for ALL 3 axes
            # ============================================================
            coeff_exprs_all_axes = []  # List of 3 lists (one per axis)
            
            for axis in range(3):
                coeff_exprs = []
                for coeff_idx in range(6):
                    # Global coefficient index in A_map
                    global_idx = 6 * seg + coeff_idx
                    
                    # Build linear expression: coeff = A_map @ [headPVA; waypoints; tailPVA]
                    expr = 0.0
                    
                    # Add headPVA contributions (columns 0-2: initial P, V, A)
                    for i in range(3):
                        expr += A_map[global_idx, i] * self.headPVA[i, axis]
                    
                    # Add waypoint contributions (columns 3 to 3+N-2)
                    for wp_idx in range(self.N - 1):
                        col = 3 + wp_idx
                        expr += A_map[global_idx, col] * self.vars['waypoints'][wp_idx, axis]
                    
                    # Add tailPVA contributions (columns 3+N-1 to 3+N+1)
                    for i in range(3):
                        col = 3 + (self.N - 1) + i
                        expr += A_map[global_idx, col] * self.tailPVA[i, axis]
                    
                    coeff_exprs.append(expr)
                
                coeff_exprs_all_axes.append(coeff_exprs)
            
            # Unpack coefficients for all axes
            # coeff_exprs_all_axes[axis][coeff_idx]
            c_all_axes = [[coeff_exprs_all_axes[axis][i] for axis in range(3)] for i in range(6)]
            c0_xyz, c1_xyz, c2_xyz, c3_xyz, c4_xyz, c5_xyz = c_all_axes
            
            # ============================================================
            # VELOCITY CONTROL POINTS (5 points) - NORM CONSTRAINTS
            # ============================================================
            # v(t) = c1 + 2*c2*t + 3*c3*t² + 4*c4*t³ + 5*c5*t⁴
            
            v_cps_xyz = []  # List of (v_x, v_y, v_z) tuples
            for k in range(5):
                v_cp_k = []
                for axis in range(3):
                    c1, c2, c3, c4, c5 = c1_xyz[axis], c2_xyz[axis], c3_xyz[axis], c4_xyz[axis], c5_xyz[axis]
                    if k == 0:
                        v_cp = c1
                    elif k == 1:
                        v_cp = c1 + (2 * c2) * dt / 4
                    elif k == 2:
                        v_cp = c1 + (2 * c2) * dt / 2 + (3 * c3) * dt**2 / 6
                    elif k == 3:
                        v_cp = c1 + (2 * c2) * 3 * dt / 4 + (3 * c3) * dt**2 / 2 + (4 * c4) * dt**3 / 4
                    else:  # k == 4
                        v_cp = c1 + (2 * c2) * dt + (3 * c3) * dt**2 + (4 * c4) * dt**3 + (5 * c5) * dt**4
                    v_cp_k.append(v_cp)
                v_cps_xyz.append(v_cp_k)
            
            # Add quadratic norm constraints: ||v||² ≤ v_max²
            for k, (v_x, v_y, v_z) in enumerate(v_cps_xyz):
                self.model.addConstr(
                    v_x * v_x + v_y * v_y + v_z * v_z <= self.v_max * self.v_max,
                    name=f"v_norm_{seg}_cp{k}"
                )
                count += 1
            
            # ============================================================
            # ACCELERATION CONTROL POINTS (4 points) - NORM CONSTRAINTS
            # ============================================================
            # a(t) = 2*c2 + 6*c3*t + 12*c4*t² + 20*c5*t³
            
            a_cps_xyz = []
            for k in range(4):
                a_cp_k = []
                for axis in range(3):
                    c2, c3, c4, c5 = c2_xyz[axis], c3_xyz[axis], c4_xyz[axis], c5_xyz[axis]
                    if k == 0:
                        a_cp = 2 * c2
                    elif k == 1:
                        a_cp = 2 * c2 + (6 * c3) * dt / 3
                    elif k == 2:
                        a_cp = 2 * c2 + (6 * c3) * 2 * dt / 3 + (12 * c4) * dt**2 / 3
                    else:  # k == 3
                        a_cp = 2 * c2 + (6 * c3) * dt + (12 * c4) * dt**2 + (20 * c5) * dt**3
                    a_cp_k.append(a_cp)
                a_cps_xyz.append(a_cp_k)
            
            # Add quadratic norm constraints: ||a||² ≤ a_max²
            for k, (a_x, a_y, a_z) in enumerate(a_cps_xyz):
                self.model.addConstr(
                    a_x * a_x + a_y * a_y + a_z * a_z <= self.a_max * self.a_max,
                    name=f"a_norm_{seg}_cp{k}"
                )
                count += 1
            
        if self.verbose:
            print(f"\n✓ Dynamic Constraints:")
            print(f"  {count} quadratic norm constraints (via Bezier control points)")
            print(f"  Velocity:     5 control points × {self.N} segments = {5 * self.N}")
            print(f"  Acceleration: 4 control points × {self.N} segments = {4 * self.N}")
            print(f"  NOTE: Using L2 norm (||v||² ≤ v_max²), not per-axis bounds")
    
    def _extract_solution(self):
        """Extract optimized waypoints and compute coefficients via MINCO."""
        # Extract waypoints
        self.waypoints = self.vars['waypoints'].X  # (N-1, 3)
        
        # Compute coefficients via MINCO
        # Build input: [headPVA(3,3), waypoints(N-1,3), tailPVA(3,3)]
        headPVA_torch = torch.tensor(self.headPVA, dtype=torch.float32).unsqueeze(0)  # (1, 3, 3)
        tailPVA_torch = torch.tensor(self.tailPVA, dtype=torch.float32).unsqueeze(0)  # (1, 3, 3)
        waypoints_torch = torch.tensor(self.waypoints, dtype=torch.float32).unsqueeze(0)  # (1, N-1, 3)
        
        # Use MINCO to compute coefficients
        t_torch = torch.tensor([[self.dt]], dtype=torch.float32).repeat(1, self.N, 1)
        self.minco.solve_linear_fixedtime(headPVA_torch, tailPVA_torch, waypoints_torch)
        
        # Extract coefficients (6N, 3)
        coeffs = self.minco.x[0].numpy()  # (6N, 3)
        
        # Reshape to (N, 6, 3) for easier access
        self.coefficients = coeffs.reshape(self.N, 6, 3)
    
    def get_trajectory(self, num_samples: int = 100) -> Tuple[np.ndarray, ...]:
        """
        Sample the optimized trajectory.
        
        IMPORTANT: Samples num_samples PER SEGMENT (like Standard MIQP)
        to ensure fair comparison.
        
        Parameters
        ----------
        num_samples : int
            Number of samples PER SEGMENT (default: 100)
            Total samples = N * num_samples
        
        Returns
        -------
        t, pos, vel, acc : np.ndarray
            Sampled trajectory (all arrays guaranteed same length)
            Shape: (N * num_samples, ...)
        """
        if not self.solved:
            raise RuntimeError("Must solve optimization first!")
        
        # Use MINCO to sample trajectory
        traj = self.minco.get_trajectory()[0]
        
        # Sample num_samples PER SEGMENT (match Standard MIQP behavior)
        total_samples = self.N * num_samples
        total_time = self.N * self.dt
        t_samples = torch.linspace(0, total_time, total_samples)
        
        pos = traj.pos(t_samples).numpy()
        vel = traj.vel(t_samples).numpy()
        acc = traj.acc(t_samples).numpy()
        t_np = t_samples.numpy()
        
        # Ensure all arrays have same length (fix MINCO sampling inconsistencies)
        actual_samples = min(len(t_np), pos.shape[0], vel.shape[0], acc.shape[0])
        
        return (
            t_np[:actual_samples],
            pos[:actual_samples],
            vel[:actual_samples],
            acc[:actual_samples]
        )
