"""
Static visualisation utilities for the SwarmPlanning pipeline.

Functions
---------
plot_environment_3d(env)
    Interactive 3-D PyVista scene — mouse zoom / pan / rotate.
"""

from __future__ import annotations

import pyvista as pv

from environment import Environment

# ─── palette ─────────────────────────────────────────────────────────────────
_CYL_COLOR   = "#4682B4"   # steelblue
_WALL_COLOR  = "#CD853F"   # peru
_FLOOR_COLOR = "#EFEFEF"


# ═════════════════════════════════════════════════════════════════════════════
# 3-D interactive view  (PyVista / html backend)
# ═════════════════════════════════════════════════════════════════════════════

def plot_environment_3d(
    env: Environment,
    jupyter_backend: str = "html",
    window_size: tuple[int, int] = (800, 600),
) -> None:
    """
    Render the environment as an interactive 3-D scene in Jupyter.

    Controls (html backend)
    -----------------------
    Left-drag     rotate  |  Right-drag / scroll   zoom  |  Middle-drag   pan

    Parameters
    ----------
    env             : Environment
    jupyter_backend : PyVista Jupyter backend.  'html' (default) embeds a
                      self-contained iframe; 'trame' uses a live server.
    window_size     : (width, height) in pixels for the embedded viewer.
    """
    w = env.world
    p = pv.Plotter(notebook=True, window_size=list(window_size))

    # ── background & lighting ─────────────────────────────────────────────────
    p.set_background("#1E1E2E")   # dark navy background

    # ── floor plane ───────────────────────────────────────────────────────────
    cx = (w.x_min + w.x_max) / 2.0
    cy = (w.y_min + w.y_max) / 2.0
    floor = pv.Plane(
        center=(cx, cy, float(w.z_min)),
        direction=(0, 0, 1),
        i_size=float(w.x_max - w.x_min),
        j_size=float(w.y_max - w.y_min),
        i_resolution=1,
        j_resolution=1,
    )
    p.add_mesh(floor, color=_FLOOR_COLOR, opacity=0.25, show_edges=False)

    # ── world bounding box (wireframe) ────────────────────────────────────────
    outline = pv.Box(bounds=(
        float(w.x_min), float(w.x_max),
        float(w.y_min), float(w.y_max),
        float(w.z_min), float(w.z_max),
    ))
    p.add_mesh(outline, color="white", opacity=0.12, style="wireframe", line_width=1)

    # ── cylinder obstacles ────────────────────────────────────────────────────
    for cyl in env.cylinders:
        mesh = pv.Cylinder(
            center=(
                float(cyl.center[0]),
                float(cyl.center[1]),
                float(w.z_min) + float(cyl.height) / 2.0,
            ),
            direction=(0, 0, 1),
            radius=float(cyl.radius),
            height=float(cyl.height),
            resolution=40,
            capping=True,
        )
        p.add_mesh(mesh, color=_CYL_COLOR, opacity=0.80, smooth_shading=True)

    # ── wall obstacles ────────────────────────────────────────────────────────
    for wall in env.walls:
        c     = wall.corners
        x_lo  = float(c[:, 0].min());  x_hi = float(c[:, 0].max())
        y_lo  = float(c[:, 1].min());  y_hi = float(c[:, 1].max())
        z_lo  = float(w.z_min);        z_hi = z_lo + float(wall.height)
        box   = pv.Box(bounds=(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi))
        p.add_mesh(box, color=_WALL_COLOR, opacity=0.85, smooth_shading=True)

    # ── axes & grid ───────────────────────────────────────────────────────────
    p.add_axes(color="white")
    p.show_bounds(
        grid="back",
        location="outer",
        ticks="outside",
        font_size=8,
        color="white",
        xtitle="X (m)",
        ytitle="Y (m)",
        ztitle="Z (m)",
    )

    # ── isometric camera ──────────────────────────────────────────────────────
    p.camera_position = "iso"
    p.reset_camera()

    p.show(jupyter_backend=jupyter_backend)
