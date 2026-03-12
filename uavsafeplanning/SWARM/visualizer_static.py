"""
Static visualisation utilities for the SwarmPlanning pipeline.

Functions
---------
plot_environment_3d(env, fleet=None)
    Interactive 3-D Plotly scene — mouse zoom / pan / rotate.
    When a Fleet is supplied, UAV start (●) and goal (◆) markers are
    overlaid, each UAV coloured from a distinct palette entry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull, HalfspaceIntersection

from environment import Environment

if TYPE_CHECKING:
    from uav import Fleet

# ─── palette ─────────────────────────────────────────────────────────────────
_CYL_COLOR      = "#4682B4"   # steelblue
_WALL_COLOR     = "#CD853F"   # peru
_FLOOR_COLOR    = "#EFEFEF"
_BG_COLOR       = "#1E1E2E"   # dark navy
_CYL_RESOLUTION = 40


# ═════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═════════════════════════════════════════════════════════════════════════════

def _cylinder_mesh(
    cx: float, cy: float,
    z_bot: float, z_top: float,
    radius: float,
    resolution: int = _CYL_RESOLUTION,
) -> tuple:
    """Return (x, y, z, i, j, k) arrays for a vertical cylinder go.Mesh3d."""
    theta = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    N = resolution

    # bottom ring (0..N-1), top ring (N..2N-1),
    # bottom center (2N), top center (2N+1)
    x = np.concatenate([cx + radius * cos_t, cx + radius * cos_t, [cx, cx]])
    y = np.concatenate([cy + radius * sin_t, cy + radius * sin_t, [cy, cy]])
    z = np.concatenate([np.full(N, z_bot), np.full(N, z_top), [z_bot, z_top]])

    ii, jj, kk = [], [], []
    for idx in range(N):
        nxt = (idx + 1) % N
        # side face (two triangles)
        ii += [idx,       nxt      ]
        jj += [N + idx,   N + idx  ]
        kk += [nxt,       N + nxt  ]
        # bottom cap
        ii.append(2 * N);     jj.append(nxt);      kk.append(idx)
        # top cap
        ii.append(2 * N + 1); jj.append(N + idx);  kk.append(N + nxt)

    return x, y, z, ii, jj, kk


def _box_mesh(
    x_lo: float, x_hi: float,
    y_lo: float, y_hi: float,
    z_lo: float, z_hi: float,
) -> tuple:
    """Return (x, y, z, i, j, k) arrays for an axis-aligned box go.Mesh3d."""
    x = [x_lo, x_hi, x_hi, x_lo, x_lo, x_hi, x_hi, x_lo]
    y = [y_lo, y_lo, y_hi, y_hi, y_lo, y_lo, y_hi, y_hi]
    z = [z_lo, z_lo, z_lo, z_lo, z_hi, z_hi, z_hi, z_hi]
    # Standard Plotly cube triangulation (12 triangles — 6 faces)
    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k = [0, 7, 2, 7, 6, 7, 1, 1, 5, 5, 7, 6]
    return x, y, z, i, j, k


def _box_wireframe(
    x_lo: float, x_hi: float,
    y_lo: float, y_hi: float,
    z_lo: float, z_hi: float,
) -> tuple:
    """Return (xs, ys, zs) for a box wireframe suitable for go.Scatter3d."""
    c = [
        (x_lo, y_lo, z_lo), (x_hi, y_lo, z_lo),
        (x_hi, y_hi, z_lo), (x_lo, y_hi, z_lo),
        (x_lo, y_lo, z_hi), (x_hi, y_lo, z_hi),
        (x_hi, y_hi, z_hi), (x_lo, y_hi, z_hi),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # bottom ring
        (4, 5), (5, 6), (6, 7), (7, 4),   # top ring
        (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
    ]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [c[a][0], c[b][0], None]
        ys += [c[a][1], c[b][1], None]
        zs += [c[a][2], c[b][2], None]
    return xs, ys, zs


# ─────────────────────────────────────────────────────────────────────────────
# Corridor helper
# ─────────────────────────────────────────────────────────────────────────────

def _corridor_polytope_trace(
    A:           np.ndarray,    # (n_c, 3) constraint normals
    b:           np.ndarray,    # (n_c,)   offsets  (A @ x <= b)
    interior_pt: np.ndarray,    # (3,)     known interior point
    color:       str,
    opacity:     float = 0.15,
    name:        str   = "",
    legendgroup: str   = "",
    showlegend:  bool  = False,
) -> Optional[go.Mesh3d]:
    """
    Convert a half-space polytope  (A @ x <= b)  to a Plotly Mesh3d trace.

    Uses scipy HalfspaceIntersection to recover the vertices, then builds
    a ConvexHull triangulation.  Returns None if the computation fails
    (degenerate polytope, numerical issues, etc.).
    """
    try:
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).ravel()

        # scipy convention: [A | -b]  so that  A@x - b <= 0  ⟺  A@x <= b
        hs     = np.hstack([A, -b[:, None]])
        hs_int = HalfspaceIntersection(hs, interior_point=interior_pt)
        verts  = np.asarray(hs_int.intersections, dtype=float)

        if verts.shape[0] < 4:
            return None

        hull = ConvexHull(verts)
        i_idx = [int(t[0]) for t in hull.simplices]
        j_idx = [int(t[1]) for t in hull.simplices]
        k_idx = [int(t[2]) for t in hull.simplices]

        return go.Mesh3d(
            x=verts[:, 0].tolist(),
            y=verts[:, 1].tolist(),
            z=verts[:, 2].tolist(),
            i=i_idx, j=j_idx, k=k_idx,
            color=color,
            opacity=opacity,
            flatshading=True,
            showscale=False,
            name=name,
            legendgroup=legendgroup,
            showlegend=showlegend,
            hoverinfo="skip",
        )
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# 3-D interactive view  (Plotly)
# ═════════════════════════════════════════════════════════════════════════════

def plot_environment_3d(
    env:         Environment,
    fleet:       Optional["Fleet"] = None,
    paths:       Optional[list]    = None,
    corridors:   Optional[list]    = None,
    window_size: tuple[int, int]   = (900, 650),
) -> None:
    """
    Render the environment as an interactive 3-D Plotly scene in Jupyter.

    Controls
    --------
    Left-drag   rotate  |  Scroll   zoom  |  Right-drag   pan

    Parameters
    ----------
    env         : Environment
    fleet       : Fleet (optional) — UAV start (●) and goal (◆) markers.
    paths       : list[PlanResult] (optional) — pruned A* paths per UAV.
                  Expected duck-type: objects with .uav_id and .path_world.
    corridors   : list[CorridorResult] (optional) — safe flight corridors per
                  UAV drawn as semi-transparent convex polytopes.
                  Expected duck-type: objects with .uav_id, .A_list, .b_list,
                  and .waypoints.
    window_size : (width, height) in pixels for the embedded viewer.
    """
    w      = env.world
    width, height = window_size
    traces = []

    # ── floor plane ───────────────────────────────────────────────────────────
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

    # ── world bounding box (wireframe) ────────────────────────────────────────
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

    # ── Safe corridors (transparent polytopes) ───────────────────────────────
    if corridors is not None and fleet is not None:
        color_map = {u.id: u.color for u in fleet.uavs}

        for cr in corridors:
            col  = color_map.get(cr.uav_id, "white")
            wpts = cr.waypoints          # (N, 3) path waypoints
            first_shown = True

            for seg_idx, (A, b) in enumerate(zip(cr.A_list, cr.b_list)):
                # Interior point = midpoint of the path segment covered by
                # this corridor (segment i spans waypoints[i] → waypoints[i+1])
                if seg_idx + 1 < len(wpts):
                    interior = 0.5 * (wpts[seg_idx] + wpts[seg_idx + 1])
                else:
                    interior = wpts[seg_idx].copy()

                trace = _corridor_polytope_trace(
                    np.asarray(A, float),
                    np.asarray(b, float).ravel(),
                    interior_pt = interior,
                    color       = col,
                    opacity     = 0.12,
                    name        = f"{cr.uav_id} — corridor",
                    legendgroup = cr.uav_id,
                    showlegend  = first_shown,
                )
                if trace is not None:
                    traces.append(trace)
                    first_shown = False

    # ── A* paths ─────────────────────────────────────────────────────────────
    if paths is not None and fleet is not None:
        # Build a colour lookup from fleet
        color_map = {u.id: u.color for u in fleet.uavs}

        for pr in paths:
            col = color_map.get(pr.uav_id, "white")
            pw  = pr.path_world   # (N, 3)

            # Path line connecting waypoints
            traces.append(go.Scatter3d(
                x=pw[:, 0].tolist(), y=pw[:, 1].tolist(), z=pw[:, 2].tolist(),
                mode="lines",
                line=dict(color=col, width=4),
                opacity=0.9,
                name=f"{pr.uav_id} — path",
                legendgroup=pr.uav_id,
                showlegend=True,
                hovertemplate=(
                    f"<b>{pr.uav_id}</b><br>"
                    "x=%{x:.1f}  y=%{y:.1f}  z=%{z:.1f}<extra></extra>"
                ),
            ))

            # Intermediate waypoint dots (skip first/last = start/goal)
            if len(pw) > 2:
                mid = pw[1:-1]
                traces.append(go.Scatter3d(
                    x=mid[:, 0].tolist(), y=mid[:, 1].tolist(), z=mid[:, 2].tolist(),
                    mode="markers",
                    marker=dict(size=5, color=col,
                                line=dict(color="white", width=1)),
                    opacity=0.8,
                    hoverinfo="skip",
                    legendgroup=pr.uav_id,
                    showlegend=False,
                ))

    # ── UAV start / goal markers ──────────────────────────────────────────────
    if fleet is not None:
        for uav in fleet.uavs:
            sx, sy, sz = float(uav.start[0]), float(uav.start[1]), float(uav.start[2])
            gx, gy, gz = float(uav.goal[0]),  float(uav.goal[1]),  float(uav.goal[2])
            col = uav.color

            # Start marker  ●
            traces.append(go.Scatter3d(
                x=[sx], y=[sy], z=[sz],
                mode="markers+text",
                marker=dict(symbol="circle", size=9, color=col,
                            line=dict(color="white", width=1.5)),
                text=[uav.id], textposition="top center",
                textfont=dict(color=col, size=10, family="monospace"),
                name=f"{uav.id} — start",
                legendgroup=uav.id,
                showlegend=True,
                hovertemplate=(
                    f"<b>{uav.id} — start</b><br>"
                    f"x={sx:.1f}  y={sy:.1f}  z={sz:.1f}<extra></extra>"
                ),
            ))

            # Goal marker  ◆
            traces.append(go.Scatter3d(
                x=[gx], y=[gy], z=[gz],
                mode="markers+text",
                marker=dict(symbol="diamond", size=9, color=col,
                            line=dict(color="white", width=1.5)),
                text=[uav.id], textposition="top center",
                textfont=dict(color=col, size=10, family="monospace"),
                name=f"{uav.id} — goal",
                legendgroup=uav.id,
                showlegend=True,
                hovertemplate=(
                    f"<b>{uav.id} — goal</b><br>"
                    f"x={gx:.1f}  y={gy:.1f}  z={gz:.1f}<extra></extra>"
                ),
            ))

            # Dashed connector line
            traces.append(go.Scatter3d(
                x=[sx, gx], y=[sy, gy], z=[sz, gz],
                mode="lines",
                line=dict(color=col, width=1.5, dash="dot"),
                opacity=0.35,
                hoverinfo="skip",
                legendgroup=uav.id,
                showlegend=False,
            ))

    # ── layout & camera ───────────────────────────────────────────────────────
    axis_style = dict(
        color="white",
        gridcolor="#333333",
        showbackground=False,
        zerolinecolor="#555555",
    )
    _ortho = dict(type="orthographic")
    _view_buttons = [
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

    _show_legend = fleet is not None and len(fleet.uavs) > 0

    fig = go.Figure(data=traces)
    fig.update_layout(
        width=width,
        height=height,
        paper_bgcolor=_BG_COLOR,
        showlegend=_show_legend,
        legend=dict(
            x=1.0, y=1.0,
            xanchor="right", yanchor="top",
            bgcolor="rgba(30,30,46,0.85)",
            bordercolor="#555577",
            borderwidth=1,
            font=dict(color="white", size=10),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
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
                buttons=_view_buttons,
            )
        ],
    )
    fig.show()
