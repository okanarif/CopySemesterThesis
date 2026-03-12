"""
Animated 3-D simulation of multi-UAV swarm trajectories.

Produces a standalone HTML file containing a Plotly animation:
  - Same dark-theme scene as visualizer_static.plot_environment_3d
    (obstacles, corridors, full trajectory lines, start/goal markers,
     XY / XZ / YZ orthographic view buttons, pan/rotate/zoom)
  - Per-frame animated elements:
      • sphere marker at each drone's current position
      • velocity-coloured trail behind each drone
      • drone marker turns crimson when inside a conflict window
  - Play / Pause buttons and a frame-scrubber slider at the bottom

Public API
----------
    from visualizer_simulation import SimulationConfig, simulate_trajectories

    cfg = SimulationConfig.from_yaml("configs/simulation.yaml")
    simulate_trajectories(env, fleet, trajectories, corridors,
                          conflict_report, cfg)
    # → writes cfg.output_html and prints the file path
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import plotly.graph_objects as go
import yaml
from scipy.interpolate import interp1d

# ── path bootstrap ─────────────────────────────────────────────────────────────
_swarm_dir   = os.path.dirname(os.path.abspath(__file__))
_uavsafe_dir = os.path.dirname(_swarm_dir)
for _p in (_swarm_dir, _uavsafe_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from environment import Environment
from logger      import get_logger
from visualizer_static import (
    _BG_COLOR,
    _cylinder_mesh,
    _box_mesh,
    _box_wireframe,
    _corridor_polytope_trace,
    _FLOOR_COLOR,
    _CYL_COLOR,
    _WALL_COLOR,
)

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationConfig:
    """
    Animation parameters loaded from ``configs/simulation.yaml``.

    Attributes
    ----------
    dt : float
        Time step between animation frames [s].
    trail_length : float
        Duration of the velocity-coloured trail shown behind each drone [s].
        0 disables the trail.
    drone_size : int
        Diameter of the sphere marker representing each drone [px].
    frame_duration_ms : int
        How long each frame is displayed during auto-play [ms].
    transition_ms : int
        Plotly transition duration between frames [ms].
        0 = crisp jumps; match frame_duration_ms for smooth interpolation.
    output_html : str
        File path (relative to the SWARM directory) for the HTML export.
    window_size : tuple[int, int]
        (width, height) of the embedded Plotly viewer [px].
    """
    dt:                float       = 0.05
    trail_length:      float       = 2.0
    drone_size:        int         = 10
    frame_duration_ms: int         = 80
    transition_ms:     int         = 0
    output_html:       str         = "simulation.html"
    window_size:       tuple       = (1100, 700)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SimulationConfig":
        """
        Load ``SimulationConfig`` from *path* (simulation.yaml).

        All keys are optional — missing values fall back to dataclass
        defaults, so the file may contain only the fields you want to
        override.

        Raises
        ------
        ValueError
            If any field fails a range check.
        """
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        dt = float(raw.get("dt", 0.05))
        if dt <= 0:
            raise ValueError(f"simulation.yaml: dt must be > 0 (got {dt})")

        trail_length = float(raw.get("trail_length", 2.0))
        if trail_length < 0:
            raise ValueError(
                f"simulation.yaml: trail_length must be >= 0 (got {trail_length})"
            )

        drone_size = int(raw.get("drone_size", 10))
        frame_duration_ms = int(raw.get("frame_duration_ms", 80))
        transition_ms     = int(raw.get("transition_ms", 0))
        output_html       = str(raw.get("output_html", "simulation.html"))

        ws_raw    = raw.get("window_size", [1100, 700])
        window_size = (int(ws_raw[0]), int(ws_raw[1]))

        return cls(
            dt                = dt,
            trail_length      = trail_length,
            drone_size        = drone_size,
            frame_duration_ms = frame_duration_ms,
            transition_ms     = transition_ms,
            output_html       = output_html,
            window_size       = window_size,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _interp_traj_at(tr, t: float) -> np.ndarray:
    """Return the (3,) world position of *tr* at time *t* (linear interp)."""
    idx  = np.searchsorted(tr.t_eval, t)
    idx  = int(np.clip(idx, 1, len(tr.t_eval) - 1))
    t0, t1 = tr.t_eval[idx - 1], tr.t_eval[idx]
    p0, p1 = tr.pos[idx - 1],    tr.pos[idx]
    if t1 == t0:
        return p0.copy()
    alpha = (t - t0) / (t1 - t0)
    return p0 + alpha * (p1 - p0)


def _interp_traj_range(tr, t_start: float, t_end: float) -> tuple:
    """
    Return (pos_arr, vel_norm_arr) for the portion of *tr* in
    [t_start, t_end].  Endpoints are linearly interpolated.

    Returns
    -------
    pos      : (M, 3) positions
    vel_norm : (M,)   speed values
    """
    mask = (tr.t_eval >= t_start) & (tr.t_eval <= t_end)
    pos  = tr.pos[mask]
    vn   = tr.vel_norm[mask]
    if len(pos) == 0:
        p = _interp_traj_at(tr, t_end)
        return p[None, :], np.array([0.0])
    return pos, vn


def _is_in_conflict(t: float, conflict_report, id_a: str, id_b: str) -> bool:
    """True if time *t* falls inside any ConflictEvent for the pair."""
    for ev in conflict_report.events_for_pair(id_a, id_b):
        if ev.t_start <= t <= ev.t_end:
            return True
    return False


def _drone_in_conflict_at(t: float, uav_id: str, conflict_report) -> bool:
    """True if *uav_id* is involved in any conflict event at time *t*."""
    for ev in conflict_report.events:
        if uav_id in (ev.uav_a, ev.uav_b):
            if ev.t_start <= t <= ev.t_end:
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Static scene builder  (same look as visualizer_static)
# ─────────────────────────────────────────────────────────────────────────────

def _build_static_traces(
    env:          Environment,
    fleet,
    corridors:    Optional[list],
    trajectories: list,
) -> list:
    """
    Build all static Plotly traces that do not change across frames:
    floor, bounding-box wireframe, obstacles, safe corridors,
    faded full-trajectory lines, and start/goal markers.
    """
    w      = env.world
    traces = []

    # ── floor ─────────────────────────────────────────────────────────────────
    fx = [w.x_min, w.x_max, w.x_max, w.x_min]
    fy = [w.y_min, w.y_min, w.y_max, w.y_max]
    fz = [w.z_min] * 4
    traces.append(go.Mesh3d(
        x=fx, y=fy, z=fz,
        i=[0, 0], j=[1, 2], k=[2, 3],
        color=_FLOOR_COLOR, opacity=0.25,
        flatshading=True, showscale=False,
        hoverinfo="skip", name="floor",
    ))

    # ── world bounding box wireframe ──────────────────────────────────────────
    wx, wy, wz = _box_wireframe(
        float(w.x_min), float(w.x_max),
        float(w.y_min), float(w.y_max),
        float(w.z_min), float(w.z_max),
    )
    traces.append(go.Scatter3d(
        x=wx, y=wy, z=wz,
        mode="lines",
        line=dict(color="white", width=1),
        opacity=0.20,
        hoverinfo="skip", name="bounds",
    ))

    # ── cylinder obstacles ────────────────────────────────────────────────────
    for cyl in env.cylinders:
        vx, vy, vz, ci, cj, ck = _cylinder_mesh(
            float(cyl.center[0]), float(cyl.center[1]),
            float(w.z_min), float(w.z_min) + float(cyl.height),
            float(cyl.radius),
        )
        traces.append(go.Mesh3d(
            x=vx, y=vy, z=vz,
            i=ci, j=cj, k=ck,
            color=_CYL_COLOR, opacity=1.0,
            flatshading=False, showscale=False,
            hoverinfo="skip", name=cyl.id,
        ))

    # ── wall obstacles ────────────────────────────────────────────────────────
    for wall in env.walls:
        c    = wall.corners
        x_lo = float(c[:, 0].min()); x_hi = float(c[:, 0].max())
        y_lo = float(c[:, 1].min()); y_hi = float(c[:, 1].max())
        z_lo = float(w.z_min);       z_hi = z_lo + float(wall.height)
        bx, by, bz, bi, bj, bk = _box_mesh(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
        traces.append(go.Mesh3d(
            x=bx, y=by, z=bz,
            i=bi, j=bj, k=bk,
            color=_WALL_COLOR, opacity=1.0,
            flatshading=True, showscale=False,
            hoverinfo="skip", name=wall.id,
        ))

    color_map = {u.id: u.color for u in fleet.uavs}

    # ── safe corridors ────────────────────────────────────────────────────────
    if corridors is not None:
        for cr in corridors:
            col   = color_map.get(cr.uav_id, "white")
            wpts  = cr.waypoints
            group = f"{cr.uav_id}_corridor"
            first = True
            for seg_idx, (A, b) in enumerate(zip(cr.A_list, cr.b_list)):
                interior = (
                    0.5 * (wpts[seg_idx] + wpts[seg_idx + 1])
                    if seg_idx + 1 < len(wpts)
                    else wpts[seg_idx].copy()
                )
                trace = _corridor_polytope_trace(
                    np.asarray(A, float),
                    np.asarray(b, float).ravel(),
                    interior_pt = interior,
                    color       = col,
                    opacity     = 0.10,
                    name        = f"{cr.uav_id} — corridor",
                    legendgroup = group,
                    showlegend  = first,
                )
                if trace is not None:
                    traces.append(trace)
                    first = False

    # ── faded full trajectory lines ───────────────────────────────────────────
    for tr in trajectories:
        col   = color_map.get(tr.uav_id, "white")
        pos   = tr.pos
        group = f"{tr.uav_id}_traj"
        traces.append(go.Scatter3d(
            x=pos[:, 0].tolist(), y=pos[:, 1].tolist(), z=pos[:, 2].tolist(),
            mode="lines",
            line=dict(color=col, width=2),
            opacity=0.20,
            hoverinfo="skip",
            legendgroup=group,
            showlegend=False,
        ))

    # ── start / goal markers ──────────────────────────────────────────────────
    for uav in fleet.uavs:
        sx, sy, sz = float(uav.start[0]), float(uav.start[1]), float(uav.start[2])
        gx, gy, gz = float(uav.goal[0]),  float(uav.goal[1]),  float(uav.goal[2])
        col = uav.color

        traces.append(go.Scatter3d(
            x=[sx], y=[sy], z=[sz],
            mode="markers+text",
            marker=dict(symbol="circle", size=14, color=col,
                        line=dict(color="white", width=1.5)),
            text=[uav.id], textposition="top center",
            textfont=dict(color=col, size=10, family="monospace"),
            name=f"{uav.id} — start",
            legendgroup=f"{uav.id}_start",
            showlegend=True,
            hovertemplate=(
                f"<b>{uav.id} — start</b><br>"
                f"x={sx:.1f}  y={sy:.1f}  z={sz:.1f}<extra></extra>"
            ),
        ))
        traces.append(go.Scatter3d(
            x=[gx], y=[gy], z=[gz],
            mode="markers+text",
            marker=dict(symbol="diamond", size=9, color=col,
                        line=dict(color="white", width=1.5)),
            text=[uav.id], textposition="top center",
            textfont=dict(color=col, size=10, family="monospace"),
            name=f"{uav.id} — goal",
            legendgroup=f"{uav.id}_goal",
            showlegend=True,
            hovertemplate=(
                f"<b>{uav.id} — goal</b><br>"
                f"x={gx:.1f}  y={gy:.1f}  z={gz:.1f}<extra></extra>"
            ),
        ))
        traces.append(go.Scatter3d(
            x=[sx, gx], y=[sy, gy], z=[sz, gz],
            mode="lines",
            line=dict(color=col, width=1.5, dash="dot"),
            opacity=0.25,
            hoverinfo="skip",
            legendgroup=f"{uav.id}_start",
            showlegend=False,
        ))

    return traces


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame animated trace builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_frame_traces(
    t:              float,
    fleet,
    trajectories:   list,
    conflict_report,
    cfg:            SimulationConfig,
) -> list:
    """
    Build the animated Plotly traces for simulation time *t*.

    Returns one pair of traces per UAV: (trail Scatter3d, drone Scatter3d).
    The ordering must match the ``n_anim_traces`` animated slots declared
    in the base figure data.
    """
    color_map   = {u.id: u.color  for u in fleet.uavs}
    v_max_fleet = max(u.v_max for u in fleet.uavs)
    traces      = []

    for tr in trajectories:
        col   = color_map.get(tr.uav_id, "white")
        t_end = min(t, tr.total_time)

        # ── trail ─────────────────────────────────────────────────────────────
        t_trail_start = max(0.0, t_end - cfg.trail_length)
        trail_pos, trail_vn = _interp_traj_range(tr, t_trail_start, t_end)

        traces.append(go.Scatter3d(
            x=trail_pos[:, 0].tolist(),
            y=trail_pos[:, 1].tolist(),
            z=trail_pos[:, 2].tolist(),
            mode="markers",
            marker=dict(
                size=3,
                color=trail_vn.tolist(),
                colorscale="Viridis",
                cmin=0.0,
                cmax=v_max_fleet,
                showscale=False,
            ),
            opacity=0.75,
            hoverinfo="skip",
            showlegend=False,
        ))

        # ── drone sphere ──────────────────────────────────────────────────────
        pos_now = _interp_traj_at(tr, t_end)
        v_now   = float(np.interp(t_end, tr.t_eval, tr.vel_norm))

        in_conflict = _drone_in_conflict_at(t, tr.uav_id, conflict_report)
        marker_color = "crimson" if in_conflict else col
        marker_line  = dict(color="white", width=2) if in_conflict else dict(color="white", width=1)

        traces.append(go.Scatter3d(
            x=[float(pos_now[0])],
            y=[float(pos_now[1])],
            z=[float(pos_now[2])],
            mode="markers+text",
            marker=dict(
                symbol="circle",
                size=cfg.drone_size,
                color=marker_color,
                line=marker_line,
            ),
            text=[f"  {tr.uav_id}  {v_now:.1f} m/s"],
            textfont=dict(color="white", size=9, family="monospace"),
            textposition="middle right",
            name=f"{tr.uav_id} — drone",
            legendgroup=f"{tr.uav_id}_drone",
            showlegend=False,
            hovertemplate=(
                f"<b>{tr.uav_id}</b><br>"
                f"t={t_end:.2f} s<br>"
                f"x={pos_now[0]:.1f}  y={pos_now[1]:.1f}  z={pos_now[2]:.1f}<br>"
                f"speed={v_now:.2f} m/s"
                + ("  ⚠ CONFLICT" if in_conflict else "")
                + "<extra></extra>"
            ),
        ))

    return traces


# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers  (same style as visualizer_static)
# ─────────────────────────────────────────────────────────────────────────────

def _build_layout(
    env:          Environment,
    t_max:        float,
    t_grid:       np.ndarray,
    cfg:          SimulationConfig,
) -> go.Layout:
    """Build the Plotly Layout with scene, camera, view buttons, and controls."""
    w             = env.world
    width, height = cfg.window_size

    axis_style = dict(
        color="white",
        gridcolor="#333333",
        showbackground=False,
        zerolinecolor="#555555",
    )
    _ortho = dict(type="orthographic")

    # ── orthographic view preset buttons ──────────────────────────────────────
    view_buttons = [
        dict(
            label="XY",
            method="relayout",
            args=["scene.camera", dict(
                eye=dict(x=0, y=0, z=2.5),
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                projection=_ortho,
            )],
        ),
        dict(
            label="XZ",
            method="relayout",
            args=["scene.camera", dict(
                eye=dict(x=0, y=-2.5, z=0),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                projection=_ortho,
            )],
        ),
        dict(
            label="YZ",
            method="relayout",
            args=["scene.camera", dict(
                eye=dict(x=2.5, y=0, z=0),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                projection=_ortho,
            )],
        ),
    ]

    # ── frame-scrubber slider ─────────────────────────────────────────────────
    # Every step gets a numeric label (just the number, no unit) so that
    # currentvalue always shows  "t = X.XX s"  even on unlabeled ticks.
    # ticklen=0 suppresses the crowded track labels; currentvalue is enough.
    slider_steps = []
    for fi, t_val in enumerate(t_grid):
        slider_steps.append(dict(
            args=[
                [f"frame_{fi}"],
                dict(
                    frame=dict(duration=cfg.frame_duration_ms, redraw=True),
                    mode="immediate",
                    transition=dict(duration=cfg.transition_ms),
                ),
            ],
            label=f"{t_val:.2f}",   # numeric label → shown by currentvalue
            method="animate",
        ))

    sliders = [dict(
        active=0,
        currentvalue=dict(
            prefix="t = ",
            suffix=" s",
            visible=True,
            xanchor="center",
            font=dict(color="white", size=14, family="monospace"),
        ),
        ticklen=0,          # hide per-step tick marks on the track
        pad=dict(t=50, b=10),
        len=0.90,
        x=0.05,
        y=0,
        bgcolor="#2A2A3E",
        bordercolor="#555577",
        font=dict(color="rgba(0,0,0,0)", size=1),   # invisible axis labels
        tickcolor="rgba(0,0,0,0)",
        steps=slider_steps,
    )]

    return go.Layout(
        width=width,
        height=height,
        paper_bgcolor=_BG_COLOR,
        showlegend=True,
        legend=dict(
            x=1.0, y=1.0,
            xanchor="right", yanchor="top",
            bgcolor="rgba(30,30,46,0.85)",
            bordercolor="#555577",
            borderwidth=1,
            font=dict(color="white", size=10),
        ),
        margin=dict(l=0, r=0, t=40, b=80),
        scene=dict(
            bgcolor=_BG_COLOR,
            xaxis=dict(title="X (m)", **axis_style),
            yaxis=dict(title="Y (m)", **axis_style),
            zaxis=dict(title="Z (m)", **axis_style),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.2),
                projection=_ortho,
            ),
        ),
        updatemenus=[
            # view preset buttons only — Play/Pause and Speed are JS-driven
            dict(
                type="buttons",
                direction="right",
                x=0.0,
                y=1.08,
                xanchor="left",
                yanchor="top",
                showactive=False,
                bgcolor="#2A2A3E",
                bordercolor="#555577",
                font=dict(color="white", size=11),
                buttons=view_buttons,
            ),
        ],
        sliders=sliders,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def simulate_trajectories(
    env:             Environment,
    fleet,
    trajectories:    list,
    corridors:       Optional[list],
    conflict_report,
    cfg:             SimulationConfig,
) -> str:
    """
    Build a Plotly animation, export it as a standalone HTML file, and
    return the absolute path to the file.

    Parameters
    ----------
    env             : Environment
    fleet           : Fleet
    trajectories    : list[TrajectoryResult]
    corridors       : list[CorridorResult] or None
    conflict_report : ConflictReport (from conflict_detector)
    cfg             : SimulationConfig

    Returns
    -------
    str
        Absolute path to the written HTML file.
    """
    n_uavs = len(fleet.uavs)

    # ── time grid ──────────────────────────────────────────────────────────────
    t_max  = max(tr.total_time for tr in trajectories)
    t_grid = np.arange(0.0, t_max + cfg.dt, cfg.dt)
    n_frames = len(t_grid)
    log.debug(f"Building animation: {n_frames} frames  dt={cfg.dt} s  t_max={t_max:.1f} s")

    # ── static traces ─────────────────────────────────────────────────────────
    static_traces = _build_static_traces(env, fleet, corridors, trajectories)
    n_static = len(static_traces)

    # ── animated traces for frame 0 ───────────────────────────────────────────
    # Each UAV contributes 2 traces: trail + drone marker.
    anim_traces_0 = _build_frame_traces(
        t_grid[0], fleet, trajectories, conflict_report, cfg
    )
    n_anim = len(anim_traces_0)   # = 2 * n_uavs

    # ── base figure ───────────────────────────────────────────────────────────
    all_traces = static_traces + anim_traces_0
    layout     = _build_layout(env, t_max, t_grid, cfg)
    fig        = go.Figure(data=all_traces, layout=layout)

    # ── build frames ──────────────────────────────────────────────────────────
    frames = []
    for fi, t_val in enumerate(t_grid):
        anim_traces = _build_frame_traces(
            t_val, fleet, trajectories, conflict_report, cfg
        )
        # Only the animated trace slots are updated per frame
        frames.append(go.Frame(
            data=anim_traces,
            name=f"frame_{fi}",
            # traces indices in fig.data that this frame updates
            traces=list(range(n_static, n_static + n_anim)),
        ))

    fig.frames = frames

    # ── JS control bar: Play/Pause toggle only ────────────────────────────────
    div_id  = "swarm_sim"
    base_ms = cfg.frame_duration_ms

    post_script = f"""
(function() {{
  var PLOT_ID   = '{div_id}';
  var BASE_MS   = {base_ms};
  var playing   = false;

  // ── button style ───────────────────────────────────────────────────────────
  var btnBase = [
    'padding:6px 16px',
    'border:1px solid #555577',
    'border-radius:6px',
    'background:#2A2A3E',
    'color:white',
    'font-size:13px',
    'font-family:monospace',
    'cursor:pointer',
    'outline:none',
    'transition:background 0.15s',
  ].join(';');

  // ── locate the plot div and make it a positioning context ─────────────────
  var gd  = document.getElementById(PLOT_ID);
  var plotDiv = gd.closest ? gd.closest('.plotly-graph-div') || gd : gd;
  plotDiv.style.position = 'relative';

  // ── Play / Pause button: positioned at same height as "t = X s" ───────────
  var ppBtn = document.createElement('button');
  ppBtn.id  = 'ppToggle';
  ppBtn.innerHTML = '&#9654;&nbsp; Play';
  ppBtn.style.cssText = btnBase + [
    'position:absolute',
    'bottom:100px',
    'left:5%',
    'z-index:9999',
    'box-shadow:0 2px 8px rgba(0,0,0,0.5)',
  ].join(';');
  plotDiv.appendChild(ppBtn);

  // ── Play / Pause logic ─────────────────────────────────────────────────────
  function startPlay() {{
    Plotly.animate(PLOT_ID, null, {{
      frame:      {{duration: BASE_MS, redraw: true}},
      transition: {{duration: 0}},
      fromcurrent: true,
      mode: 'immediate',
    }});
    ppBtn.innerHTML = '&#9646;&#9646;&nbsp; Pause';
    playing = true;
  }}

  function doPause() {{
    Plotly.animate(PLOT_ID, [null], {{
      frame: {{duration: 0, redraw: false}},
      mode:  'immediate',
    }});
    ppBtn.innerHTML = '&#9654;&nbsp; Play';
    playing = false;
  }}

  ppBtn.addEventListener('click', function() {{
    playing ? doPause() : startPlay();
  }});

  // when animation reaches the last frame Plotly fires 'plotly_animated'
  gd.on('plotly_animated', function() {{
    ppBtn.innerHTML = '&#9654;&nbsp; Play';
    playing = false;
  }});
}})();
"""

    # ── export ────────────────────────────────────────────────────────────────
    out_path = os.path.join(_swarm_dir, cfg.output_html)
    fig.write_html(
        out_path,
        div_id=div_id,
        include_plotlyjs="cdn",
        full_html=True,
        auto_open=False,
        post_script=post_script,
        config=dict(
            responsive=True,
            displayModeBar=True,
            modeBarButtonsToRemove=["resetCameraLastSave3d"],
        ),
    )

    log.debug(f"Simulation HTML written → {out_path}")
    return out_path
