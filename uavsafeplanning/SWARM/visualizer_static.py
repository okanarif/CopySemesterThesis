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


# ═════════════════════════════════════════════════════════════════════════════
# 3-D interactive view  (Plotly)
# ═════════════════════════════════════════════════════════════════════════════

def plot_environment_3d(
    env: Environment,
    fleet: Optional["Fleet"] = None,
    window_size: tuple[int, int] = (900, 650),
) -> None:
    """
    Render the environment as an interactive 3-D Plotly scene in Jupyter.

    Controls
    --------
    Left-drag   rotate  |  Scroll   zoom  |  Right-drag   pan

    Parameters
    ----------
    env         : Environment
    fleet       : Fleet (optional) — when provided, UAV start/goal markers
                  are drawn. Each UAV gets a distinct colour, a filled circle
                  for its start (●) and a diamond for its goal (◆).
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
