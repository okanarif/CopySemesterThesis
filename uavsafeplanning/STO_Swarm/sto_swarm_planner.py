"""
STO (Spatio-Temporal Optimizer) Trajectory Planner
Gradient-based optimization with MINCO representation and soft constraints.
"""

import torch
import numpy as np
import time as time_module
from scipy.interpolate import interp1d
from traj_gen_utils import MINCO_S3NU
from sto_swarm_utils import tau_to_time, time_to_tau


class STO_Swarm_Planner:
    """
    Spatio-Temporal Optimizer for trajectory planning.
    
    Uses gradient-based optimization (L-BFGS) with:
    - MINCO-based trajectory representation (5th-order polynomials)
    - Direct waypoint optimization (no xi diffeomorphism)
    - Soft constraints (penalties for violations)
    - Variable time allocation
    
    Key Features:
    - Minimizes jerk energy (primary objective)
    - Soft penalties for velocity, acceleration, corridor violations
    - Selectable corridor cost type: L1, L2, or Log(1+v)
    - Fast convergence with L-BFGS optimizer
    - Optional adaptive weight escalation (sequential penalty method)
    """
    
    def __init__(
        self,
        n_segments: int,
        A_list: list,
        b_list: list,
        waypoints_init: np.ndarray,
        time_init: np.ndarray,
        v_max: float,
        a_max: float,
        lambda_jerk: float = 10.0,
        lambda_time: float = 0.05,
        lambda_vel: float = 100.0,
        lambda_acc: float = 5.0,
        lambda_corridor: float = 1000.0,
        corridor_cost_type: str = 'l2',
        learn_rate: float = 0.05,
        max_iter: int = 50,
        verbose: bool = True,
        adaptive_weights: bool = False,
        weight_phases: int = 3,
        weight_scale_factor: float = 5.0,
        frozen_trajectories: list = None,
        lambda_sep: float = 0.0,
        temporal_only: bool = False,
    ):
        """
        Initialize STO planner.
        
        Parameters
        ----------
        n_segments : int
            Number of trajectory segments
        A_list : list
            List of A matrices for corridor constraints (H-representation)
        b_list : list
            List of b vectors for corridor constraints
        waypoints_init : np.ndarray
            Initial waypoints (n_waypoints, 3)
        time_init : np.ndarray
            Initial time allocation per segment (n_segments,)
        v_max : float
            Maximum velocity (m/s)
        a_max : float
            Maximum acceleration (m/s²)
        lambda_jerk : float
            Weight for jerk energy (primary objective, constant across phases)
        lambda_time : float
            Weight for total time (constant across phases)
        lambda_vel : float
            Weight for velocity violation penalty (target value for final phase)
        lambda_acc : float
            Weight for acceleration violation penalty (target value for final phase)
        lambda_corridor : float
            Weight for corridor violation penalty (target value for final phase)
        corridor_cost_type : str
            Corridor penalty formulation: 'l2' (quadratic), 'l1' (linear),
            or 'log' (log(1 + violation), slower growth than L1 for large violations)
        learn_rate : float
            Learning rate for L-BFGS optimizer
        max_iter : int
            Maximum optimization iterations (total, spread across phases)
        verbose : bool
            Print optimization progress
        adaptive_weights : bool
            Enable sequential penalty escalation. When True, constraint weights
            (corridor, vel, acc) start at lambda / scale^(phases-1) and are
            multiplied by weight_scale_factor at each phase boundary, reaching
            their final target values in the last phase. The L-BFGS Hessian
            approximation is reset at each phase transition.
        weight_phases : int
            Number of escalation phases (only used if adaptive_weights=True)
        weight_scale_factor : float
            Multiplicative factor applied to constraint weights at each phase
            boundary (only used if adaptive_weights=True)
        frozen_trajectories : list, optional
            List of (t_eval_np, pos_np, safety_dist) tuples representing
            trajectories of other UAVs that this planner must avoid.
            Used during conflict-resolution replanning.
        lambda_sep : float
            Weight for inter-UAV separation penalty. Only active when
            frozen_trajectories is provided.
        temporal_only : bool
            When True, freeze waypoint positions and only optimise the
            time allocation (tau).  Used for temporal de-confliction.
        """
        self.n_segments = n_segments
        self.n_waypoints = waypoints_init.shape[0]
        self.v_max = v_max
        self.a_max = a_max
        self.verbose = verbose
        self.max_iter = max_iter
        self._learn_rate = learn_rate
        self.temporal_only = temporal_only
        self.lambda_sep = lambda_sep
        
        # Build frozen trajectory interpolators for separation penalty
        self._frozen_interps = []
        if frozen_trajectories:
            for (t_eval_np, pos_np, safety_d) in frozen_trajectories:
                interp_fns = []
                for k in range(3):
                    f = interp1d(
                        t_eval_np, pos_np[:, k],
                        kind='linear', bounds_error=False,
                        fill_value=(pos_np[0, k], pos_np[-1, k]),
                    )
                    interp_fns.append(f)
                self._frozen_interps.append((interp_fns, float(safety_d)))
        
        # Adaptive weight settings
        self.adaptive_weights = adaptive_weights
        self.weight_phases = weight_phases if adaptive_weights else 1
        self.weight_scale_factor = weight_scale_factor
        
        # Store final (target) weights — these are what the user specified
        self._lambda_vel_final      = lambda_vel
        self._lambda_acc_final      = lambda_acc
        self._lambda_corridor_final = lambda_corridor
        
        # Objective weights (jerk & time never change)
        self.lambda_jerk = lambda_jerk
        self.lambda_time = lambda_time
        
        # Constraint weights: start low when adaptive, start at final when not
        if adaptive_weights and weight_phases > 1:
            divisor = weight_scale_factor ** (weight_phases - 1)
            self.lambda_vel      = lambda_vel      / divisor
            self.lambda_acc      = lambda_acc      / divisor
            self.lambda_corridor = lambda_corridor / divisor
        else:
            self.lambda_vel      = lambda_vel
            self.lambda_acc      = lambda_acc
            self.lambda_corridor = lambda_corridor
        
        corridor_cost_type = corridor_cost_type.lower()
        if corridor_cost_type not in ('l1', 'l2', 'log'):
            raise ValueError(f"corridor_cost_type must be 'l1', 'l2', or 'log', got '{corridor_cost_type}'")
        self.corridor_cost_type = corridor_cost_type
        
        # Convert corridor constraints to tensors
        self.A_tensors = [torch.tensor(A, dtype=torch.float32) for A in A_list]
        self.b_tensors = [torch.tensor(b, dtype=torch.float32) for b in b_list]
        
        # Initialize waypoints
        self.waypoints = torch.tensor(
            [waypoints_init], dtype=torch.float32,
            requires_grad=not temporal_only,
        )
        
        # Initialize time allocation (unconstrained tau parameterization)
        time_init_tensor = torch.tensor(time_init, dtype=torch.float32).view(1, n_segments, 1)
        tau_init = time_to_tau(time_init_tensor)
        self.tau = tau_init.clone().detach().requires_grad_(True)
        
        # MINCO model
        self.model = MINCO_S3NU(n_segments)
        
        # Optimizer: temporal_only → only tau; otherwise both
        opt_params = [self.tau] if temporal_only else [self.waypoints, self.tau]
        self.optimizer = torch.optim.LBFGS(
            opt_params,
            lr=learn_rate,
            line_search_fn="strong_wolfe"
        )
        
        # Storage
        self.cost_history = {
            "total": [],
            "jerk": [],
            "time": [],
            "vel": [],
            "acc": [],
            "corridor": [],
            "separation": [],
        }
        # Weight history: records phase snapshot at every phase boundary
        self.weight_history = []
        self.solved = False
        self.runtime = 0.0
        
    def solve(
        self,
        pos_init: np.ndarray,
        vel_init: np.ndarray,
        acc_init: np.ndarray,
        pos_final: np.ndarray,
        vel_final: np.ndarray,
        acc_final: np.ndarray
    ):
        """
        Solve trajectory optimization problem.
        
        Parameters
        ----------
        pos_init : np.ndarray
            Initial position (3,)
        vel_init : np.ndarray
            Initial velocity (3,)
        acc_init : np.ndarray
            Initial acceleration (3,)
        pos_final : np.ndarray
            Final position (3,)
        vel_final : np.ndarray
            Final velocity (3,)
        acc_final : np.ndarray
            Final acceleration (3,)
        """
        if self.verbose:
            print("=" * 60)
            if self.temporal_only:
                print("STO TEMPORAL REPLAN (waypoints frozen)")
            else:
                print("STO TRAJECTORY OPTIMIZATION")
            print("=" * 60)
            print(f"Segments:       {self.n_segments}")
            print(f"Waypoints:      {self.n_waypoints}")
            print(f"Max iterations: {self.max_iter}")
            print(f"Corridor cost:  {self.corridor_cost_type.upper()} penalty")
            if self.adaptive_weights:
                print(f"Adaptive weights: ON  "
                      f"({self.weight_phases} phases, ×{self.weight_scale_factor:.1f} per phase)")
            else:
                print(f"Adaptive weights: OFF (fixed weights)")
            print(f"\nObjective weights (fixed):")
            print(f"  λ_jerk     = {self.lambda_jerk:.2f}")
            print(f"  λ_time     = {self.lambda_time:.2f}")
            print(f"\nConstraint weights (phase 1 / final target):")
            print(f"  λ_vel      = {self.lambda_vel:.2f}  →  {self._lambda_vel_final:.2f}")
            print(f"  λ_acc      = {self.lambda_acc:.2f}  →  {self._lambda_acc_final:.2f}")
            print(f"  λ_corridor = {self.lambda_corridor:.2f}  →  {self._lambda_corridor_final:.2f}"
                  f"  [{self.corridor_cost_type.upper()}]")
            if self._frozen_interps:
                print(f"\nSeparation penalty:")
                print(f"  λ_sep      = {self.lambda_sep:.2f}")
                print(f"  frozen trajectories: {len(self._frozen_interps)}")
        
        # Store boundary conditions (fixed, not optimized)
        self.headPVA_pos = torch.tensor([[pos_init]], dtype=torch.float32)
        self.headPVA_vel = torch.tensor([[vel_init]], dtype=torch.float32)
        self.headPVA_acc = torch.tensor([[acc_init]], dtype=torch.float32)
        self.tailPVA_pos = torch.tensor([[pos_final]], dtype=torch.float32)
        self.tailPVA_vel = torch.tensor([[vel_final]], dtype=torch.float32)
        self.tailPVA_acc = torch.tensor([[acc_final]], dtype=torch.float32)
        
        # Start timer
        start_time = time_module.time()
        
        def closure():
            """Compute objective and gradients"""
            self.optimizer.zero_grad()
            
            # 1) Convert tau to positive time durations (clamped to prevent
            #    degenerate solutions where segments grow unbounded)
            t = tau_to_time(self.tau).clamp(min=0.01, max=100.0)
            
            # 2) Build boundary conditions
            headPVA = torch.cat([self.headPVA_pos, self.headPVA_vel, self.headPVA_acc], dim=1)
            tailPVA = torch.cat([self.tailPVA_pos, self.tailPVA_vel, self.tailPVA_acc], dim=1)
            
            # 3) MINCO: Compute polynomial coefficients and jerk energy
            jerk_energy = self.model(headPVA, tailPVA, self.waypoints, t)
            traj = self.model.get_trajectory()[0]
            
            # 4) Sample trajectory for constraint checking
            T_total = t.sum()
            n_samples = 100
            t_samples = torch.linspace(0.0, T_total.item(), steps=n_samples, device=t.device)
            
            pos = traj.pos(t_samples)
            vel = traj.vel(t_samples)
            acc = traj.acc(t_samples)
            
            vel_norm = vel.norm(dim=-1)
            acc_norm = acc.norm(dim=-1)
            
            # 5) Velocity and acceleration penalties
            vel_violation = torch.clamp(vel_norm - self.v_max, min=0.0)
            acc_violation = torch.clamp(acc_norm - self.a_max, min=0.0)
            
            cost_vel = (vel_violation ** 2).mean()
            cost_acc = (acc_violation ** 2).mean()
            
            
            # 6) Corridor penalty (soft constraint)
            t_flat = t.view(-1)
            cum_t = torch.cumsum(t_flat, dim=0)
            seg_idx = torch.searchsorted(cum_t, t_samples)
            
            cost_corridor = torch.zeros(1, device=t.device)
            actual_samples = pos.shape[0]
            
            for s_idx in range(actual_samples):
                if s_idx >= len(seg_idx):
                    break
                
                seg = int(seg_idx[s_idx].item())
                if seg >= self.n_segments:
                    seg = self.n_segments - 1
                
                # Map segment index to corridor index (when n_segments > n_corridors)
                n_corridors = len(self.A_tensors)
                corridor_idx = min((seg * n_corridors) // self.n_segments, n_corridors - 1)
                
                A_seg = self.A_tensors[corridor_idx].to(t.device)
                b_seg = self.b_tensors[corridor_idx].to(t.device).view(-1)
                
                pos_sample = pos[s_idx]
                violation = A_seg @ pos_sample - b_seg
                violation_positive = torch.clamp(violation, min=0.0)
                
                if self.corridor_cost_type == 'l1':
                    cost_corridor = cost_corridor + violation_positive.sum()
                elif self.corridor_cost_type == 'l2':
                    cost_corridor = cost_corridor + (violation_positive ** 2).sum()
                else:  # 'log': log(1 + violation) — smooth, slower growth than L1
                    cost_corridor = cost_corridor + torch.log1p(violation_positive).sum()
            
            cost_corridor = cost_corridor / actual_samples
            
            # 7) Inter-UAV separation penalty (active during replanning)
            cost_sep = torch.zeros(1, device=t.device)
            if self._frozen_interps and self.lambda_sep > 0:
                t_np = t_samples[:actual_samples].detach().cpu().numpy()
                for (interp_fns, safety_d) in self._frozen_interps:
                    other_pos_np = np.stack(
                        [f(t_np) for f in interp_fns], axis=-1
                    )
                    other_pos = torch.tensor(
                        other_pos_np, dtype=torch.float32, device=t.device,
                    )
                    dist = (pos[:actual_samples] - other_pos).norm(dim=-1)
                    viol = torch.clamp(safety_d - dist, min=0.0)
                    cost_sep = cost_sep + (viol ** 2).mean()
            
            # 8) Total weighted cost
            cost_jerk = (jerk_energy if hasattr(jerk_energy, 'item')
                         else torch.tensor(float(jerk_energy)))
            cost_time = T_total

            total_cost = (self.lambda_jerk * cost_jerk +
                         self.lambda_time * cost_time +
                         self.lambda_vel * cost_vel +
                         self.lambda_acc * cost_acc +
                         self.lambda_corridor * cost_corridor +
                         self.lambda_sep * cost_sep)

            # 9) Backward pass
            total_cost.backward()

            # Store costs
            self.cost_history["total"].append(total_cost.item())
            self.cost_history["jerk"].append(cost_jerk.item())
            self.cost_history["time"].append(cost_time.item())
            self.cost_history["vel"].append(cost_vel.item())
            self.cost_history["acc"].append(cost_acc.item())
            self.cost_history["corridor"].append(cost_corridor.item())
            self.cost_history["separation"].append(cost_sep.item())
            
            return total_cost
        
        # ── Optimization loop (with optional phase-based weight escalation) ───────
        iters_per_phase = max(1, self.max_iter // self.weight_phases)
        
        _has_sep = bool(self._frozen_interps and self.lambda_sep > 0)
        if self.verbose:
            hdr = (f"\n{'Iter':>4} | {'Phase':>5} | {'Total':>12} | {'Jerk':>10} | "
                   f"{'Time':>8} | {'Vel':>10} | {'Acc':>10} | {'Corr':>10}")
            if _has_sep:
                hdr += f" | {'Sep':>10}"
            print(hdr)
            print("-" * (110 if _has_sep else 98))
        
        global_iter = 0
        for phase in range(self.weight_phases):
            # ── Phase transition: scale up constraint weights and reset L-BFGS ──
            if phase > 0 and self.adaptive_weights:
                self.lambda_vel = min(
                    self.lambda_vel * self.weight_scale_factor,
                    self._lambda_vel_final
                )
                self.lambda_acc = min(
                    self.lambda_acc * self.weight_scale_factor,
                    self._lambda_acc_final
                )
                self.lambda_corridor = min(
                    self.lambda_corridor * self.weight_scale_factor,
                    self._lambda_corridor_final
                )
                # Reset L-BFGS: old Hessian approximation is based on previous
                # cost landscape and becomes misleading after a weight change.
                opt_params = [self.tau] if self.temporal_only else [self.waypoints, self.tau]
                self.optimizer = torch.optim.LBFGS(
                    opt_params,
                    lr=self._learn_rate,
                    line_search_fn="strong_wolfe"
                )
                if self.verbose:
                    print(f"\n  ── Phase {phase + 1}/{self.weight_phases}: "
                          f"λ_vel={self.lambda_vel:.1f}  "
                          f"λ_acc={self.lambda_acc:.1f}  "
                          f"λ_corridor={self.lambda_corridor:.1f}  "
                          f"(L-BFGS reset) ──")
            
            # Record weight snapshot at start of this phase
            self.weight_history.append({
                "phase":           phase + 1,
                "iter_start":      global_iter,
                "lambda_vel":      self.lambda_vel,
                "lambda_acc":      self.lambda_acc,
                "lambda_corridor": self.lambda_corridor,
            })
            
            # ── Iterations for this phase ─────────────────────────────────────
            # Last phase gets any remaining iterations due to integer division
            phase_iters = (
                iters_per_phase
                if phase < self.weight_phases - 1
                else self.max_iter - global_iter
            )
            
            for _ in range(phase_iters):
                self.optimizer.step(closure)
                global_iter += 1
                
                if self.verbose and (global_iter % 5 == 0 or global_iter == 1):
                    row = (f"{global_iter:4d} | {phase+1:5d} | "
                           f"{self.cost_history['total'][-1]:12.6f} | "
                           f"{self.cost_history['jerk'][-1]:10.6f} | "
                           f"{self.cost_history['time'][-1]:8.2f} | "
                           f"{self.cost_history['vel'][-1]:10.6f} | "
                           f"{self.cost_history['acc'][-1]:10.6f} | "
                           f"{self.cost_history['corridor'][-1]:10.6f}")
                    if _has_sep:
                        row += f" | {self.cost_history['separation'][-1]:10.6f}"
                    print(row)
        
        # Stop timer
        self.runtime = time_module.time() - start_time
        self.solved = True
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("✓ OPTIMIZATION COMPLETE!")
            print("=" * 60)
            print(f"Runtime:                  {self.runtime:.3f} seconds")
            print(f"Final jerk cost:          {self.cost_history['jerk'][-1]:.6f}")
            print(f"Final corridor violation: {self.cost_history['corridor'][-1]:.6f}")
            if _has_sep:
                print(f"Final separation cost:    {self.cost_history['separation'][-1]:.6f}")
            if self.adaptive_weights:
                print(f"Phases completed:         {self.weight_phases}")
                print(f"Final λ_vel:              {self.lambda_vel:.2f}")
                print(f"Final λ_acc:              {self.lambda_acc:.2f}")
                print(f"Final λ_corridor:         {self.lambda_corridor:.2f}")
            print("=" * 60)
    
    def get_results(self, n_samples: int = 200):
        """
        Extract optimized trajectory and compute statistics.
        
        Parameters
        ----------
        n_samples : int
            Number of samples for trajectory evaluation
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'waypoints': optimized waypoints (n_waypoints, 3)
            - 'time_allocation': time per segment (n_segments,)
            - 'total_time': total trajectory time
            - 'pos': sampled positions (n_samples, 3)
            - 'vel': sampled velocities (n_samples, 3)
            - 'acc': sampled accelerations (n_samples, 3)
            - 'jerk': sampled jerks (n_samples, 3)
            - 't_eval': time samples (n_samples,)
            - 'violations': dict with max violations
            - 'runtime': optimization time (seconds)
            - 'cost_history': optimization history
            - 'jerk_cost': final jerk cost
        """
        if not self.solved:
            raise RuntimeError("Trajectory not solved yet. Call solve() first.")
        
        # Extract optimized parameters (apply same clamp as closure)
        waypoints_opt = self.waypoints.detach().numpy()[0]
        time_opt = tau_to_time(self.tau).clamp(min=0.01, max=100.0).detach().numpy()[0, :, 0]
        total_time = time_opt.sum()
        
        # Get final trajectory
        with torch.no_grad():
            t_final = tau_to_time(self.tau).clamp(min=0.01, max=100.0)
            headPVA = torch.cat([self.headPVA_pos, self.headPVA_vel, self.headPVA_acc], dim=1)
            tailPVA = torch.cat([self.tailPVA_pos, self.tailPVA_vel, self.tailPVA_acc], dim=1)
            
            _ = self.model(headPVA, tailPVA, self.waypoints, t_final)
            traj = self.model.get_trajectory()[0]
        
        # Sample trajectory
        t_eval = torch.linspace(0.0, total_time, steps=n_samples)
        pos = traj.pos(t_eval).numpy()
        vel = traj.vel(t_eval).numpy()
        acc = traj.acc(t_eval).numpy()
        jerk = traj.jerk(t_eval).numpy()
        
        # IMPORTANT: Ensure all arrays have same length
        # (MINCO may return fewer points due to endpoint/precision issues)
        actual_samples = min(pos.shape[0], vel.shape[0], acc.shape[0], jerk.shape[0])
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
        
        max_vel_viol = float(vel_violations.max()) if actual_samples > 0 else 0.0
        max_acc_viol = float(acc_violations.max()) if actual_samples > 0 else 0.0
        
        # Check corridor violations
        corridor_viols = []
        n_corridors = len(self.A_tensors)
        
        for i in range(actual_samples):
            pos_sample = pos[i]
            t_sample = t_eval[i].item()
            t_cumsum = np.cumsum(time_opt)
            seg = np.searchsorted(t_cumsum, t_sample)
            
            # Clamp segment index to valid range
            if seg >= self.n_segments:
                seg = self.n_segments - 1
            
            # Use same proportional mapping as closure() to stay consistent
            corridor_idx = min((seg * n_corridors) // self.n_segments, n_corridors - 1)
            
            A_seg = self.A_tensors[corridor_idx].numpy()
            b_seg = self.b_tensors[corridor_idx].numpy().flatten()
            
            violation = A_seg @ pos_sample - b_seg
            max_viol = violation.max()
            if max_viol > 0:
                corridor_viols.append(max_viol)
        
        max_corr_viol = max(corridor_viols) if len(corridor_viols) > 0 else 0.0
        
        return {
            'waypoints': waypoints_opt,
            'time_allocation': time_opt,
            'total_time': total_time,
            'pos': pos,
            'vel': vel,
            'acc': acc,
            'jerk': jerk,
            'vel_norm': vel_norm,
            'acc_norm': acc_norm,
            'jerk_norm': jerk_norm,
            't_eval': t_eval.numpy(),
            'violations': {
                'vel': max_vel_viol,
                'acc': max_acc_viol,
                'corridor': max_corr_viol
            },
            'runtime': self.runtime,
            'cost_history': self.cost_history,
            'weight_history': self.weight_history,
            'jerk_cost': self.cost_history['jerk'][-1]
        }
