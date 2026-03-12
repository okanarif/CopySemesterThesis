"""
Environment model for multi-UAV trajectory planning.

Provides:
  - WorldBounds      : axis-aligned 3-D planning space + world ↔ voxel index conversion
  - CylinderObstacle : vertical cylindrical column
  - WallObstacle     : extruded rectangular footprint (used for narrow passages)
  - ConfigValidationError : raised on fatal YAML errors
  - Environment      : loads YAML, validates, exposes analytic collision check
                       and rasterised VoxelGrid for A*

Typical usage
-------------
    from environment import Environment, ConfigValidationError

    try:
        env = Environment.from_yaml("configs/environment.yaml")
    except ConfigValidationError as e:
        ...

    vg    = env.to_voxel_grid()          # for A*
    hit   = env.is_point_in_obstacle(p)  # analytic check
    print(env.summary())
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import yaml

# ── add traj_gen_utils to sys.path ────────────────────────────────────────────
_swarm_dir      = os.path.dirname(os.path.abspath(__file__))
_uavsafe_dir    = os.path.dirname(_swarm_dir)
for _p in (_swarm_dir, _uavsafe_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from traj_gen_utils import VoxelGrid
from traj_gen_utils.voxel_grid import GridInfo
from logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions & warning carrier
# ─────────────────────────────────────────────────────────────────────────────

class ConfigValidationError(Exception):
    """Raised when the environment YAML has an unrecoverable error."""


class _ConfigWarning:
    __slots__ = ("field", "message")

    def __init__(self, field: str, message: str) -> None:
        self.field   = field
        self.message = message


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers (polygon containment)
# ─────────────────────────────────────────────────────────────────────────────

def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Scalar ray-casting point-in-polygon test (2D).

    Parameters
    ----------
    point   : (2,) array   [x, y]
    polygon : (N, 2) array — vertices in order (open or closed)

    Returns
    -------
    bool
    """
    x, y   = float(point[0]), float(point[1])
    n      = len(polygon)
    inside = False
    j      = n - 1
    for i in range(n):
        xi, yi = float(polygon[i, 0]), float(polygon[i, 1])
        xj, yj = float(polygon[j, 0]), float(polygon[j, 1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    Vectorised ray-casting test for many 2-D points.

    Parameters
    ----------
    points  : (N, 2) array
    polygon : (M, 2) array

    Returns
    -------
    (N,) bool array
    """
    x      = points[:, 0]
    y      = points[:, 1]
    n      = len(polygon)
    inside = np.zeros(len(points), dtype=bool)
    j      = n - 1
    for i in range(n):
        xi, yi = float(polygon[i, 0]), float(polygon[i, 1])
        xj, yj = float(polygon[j, 0]), float(polygon[j, 1])
        # Edges where yi == yj are horizontal; the straddle test is already
        # False for those rows, so the x-intersection value is never used.
        # Use np.where to avoid the divide-by-zero RuntimeWarning.
        denom    = yj - yi
        safe_den = np.where(denom == 0.0, 1.0, denom)   # avoid /0 in np
        x_inter  = (xj - xi) * (y - yi) / safe_den + xi
        straddle = (yi > y) != (yj > y)
        crossing = straddle & (x < x_inter)
        inside  ^= crossing
        j = i
    return inside


# ─────────────────────────────────────────────────────────────────────────────
# World bounds
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WorldBounds:
    """Axis-aligned 3-D planning space with a uniform voxel resolution."""

    x_range:    Tuple[float, float]
    y_range:    Tuple[float, float]
    z_range:    Tuple[float, float]
    resolution: float

    # ── convenience properties ────────────────────────────────────────────────

    @property
    def x_min(self) -> float: return self.x_range[0]
    @property
    def x_max(self) -> float: return self.x_range[1]
    @property
    def y_min(self) -> float: return self.y_range[0]
    @property
    def y_max(self) -> float: return self.y_range[1]
    @property
    def z_min(self) -> float: return self.z_range[0]
    @property
    def z_max(self) -> float: return self.z_range[1]

    @property
    def nx(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def ny(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))

    @property
    def nz(self) -> int:
        return int(round((self.z_max - self.z_min) / self.resolution))

    # ── coordinate conversions ────────────────────────────────────────────────

    def world_to_index(self, point: np.ndarray) -> Tuple[int, int, int]:
        """Convert a world-space point [x, y, z] to integer voxel indices."""
        return (
            int((point[0] - self.x_min) / self.resolution),
            int((point[1] - self.y_min) / self.resolution),
            int((point[2] - self.z_min) / self.resolution),
        )

    def index_to_world(self, idx: Tuple[int, int, int]) -> np.ndarray:
        """Return the centre of voxel (i, j, k) in world coordinates."""
        return np.array([
            self.x_min + (idx[0] + 0.5) * self.resolution,
            self.y_min + (idx[1] + 0.5) * self.resolution,
            self.z_min + (idx[2] + 0.5) * self.resolution,
        ])

    def in_bounds(self, point: np.ndarray) -> bool:
        """Return True if world point lies inside the planning volume."""
        return (
            self.x_min <= point[0] <= self.x_max and
            self.y_min <= point[1] <= self.y_max and
            self.z_min <= point[2] <= self.z_max
        )


# ─────────────────────────────────────────────────────────────────────────────
# Obstacle types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CylinderObstacle:
    """
    Vertical cylindrical obstacle.

    Occupies all points (x, y, z) such that
        (x - cx)² + (y - cy)² ≤ radius²   and   z_floor ≤ z ≤ z_floor + height
    where z_floor = world.z_min.
    """

    id:     str
    center: np.ndarray   # shape (2,) — [x, y]
    radius: float
    height: float

    def contains_point(self, point: np.ndarray, z_floor: float = 0.0) -> bool:
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        return (
            dx * dx + dy * dy <= self.radius ** 2
            and z_floor <= point[2] <= z_floor + self.height
        )


@dataclass
class WallObstacle:
    """
    Rectangular wall obstacle — extruded polygon footprint.

    The footprint is defined by exactly 4 (x, y) corner vertices.
    Occupies all points inside the polygon (XY) up to z_floor + height.
    """

    id:      str
    corners: np.ndarray   # shape (4, 2) — [[x0,y0], …, [x3,y3]]
    height:  float

    def contains_point_xy(self, point: np.ndarray) -> bool:
        return _point_in_polygon(point[:2], self.corners)

    def contains_point(self, point: np.ndarray, z_floor: float = 0.0) -> bool:
        return (
            z_floor <= point[2] <= z_floor + self.height
            and self.contains_point_xy(point)
        )


# ─────────────────────────────────────────────────────────────────────────────
# YAML validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_float(value: object, path: str) -> float:
    """Cast to float or raise ConfigValidationError."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ConfigValidationError(
            f"[{path}] Expected a number, got {type(value).__name__!r}: {value!r}"
        )


def _validate_world(raw: dict, warnings: List[_ConfigWarning]) -> WorldBounds:
    for key in ("x_range", "y_range", "z_range", "resolution"):
        if key not in raw:
            raise ConfigValidationError(f"[world] Missing required field: '{key}'")

    ranges: dict = {}
    for axis in ("x_range", "y_range", "z_range"):
        rng = raw[axis]
        if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
            raise ConfigValidationError(
                f"[world.{axis}] Must be a list of 2 numbers, got: {rng!r}"
            )
        lo = _require_float(rng[0], f"world.{axis}[0]")
        hi = _require_float(rng[1], f"world.{axis}[1]")
        if lo >= hi:
            raise ConfigValidationError(
                f"[world.{axis}] min ({lo}) must be strictly less than max ({hi})"
            )
        ranges[axis] = (lo, hi)

    res = _require_float(raw["resolution"], "world.resolution")
    if res <= 0.0:
        raise ConfigValidationError(
            f"[world.resolution] Must be > 0, got: {res}"
        )
    if res > 5.0:
        warnings.append(_ConfigWarning(
            "world.resolution",
            f"Resolution {res} m/voxel is very coarse. A* path quality may suffer."
        ))

    return WorldBounds(
        x_range=ranges["x_range"],
        y_range=ranges["y_range"],
        z_range=ranges["z_range"],
        resolution=res,
    )


def _validate_cylinder(
    raw: dict,
    idx: int,
    world: WorldBounds,
    warnings: List[_ConfigWarning],
) -> CylinderObstacle:
    obs_id = str(raw.get("id", f"cylinder_{idx}"))
    pfx    = f"obstacles.cylinders[{idx}] '{obs_id}'"

    for key in ("center", "radius", "height"):
        if key not in raw:
            raise ConfigValidationError(f"[{pfx}] Missing required field: '{key}'")

    center = raw["center"]
    if not (isinstance(center, (list, tuple)) and len(center) == 2):
        raise ConfigValidationError(
            f"[{pfx}] 'center' must be [x, y], got: {center!r}"
        )
    cx = _require_float(center[0], f"{pfx}.center[0]")
    cy = _require_float(center[1], f"{pfx}.center[1]")

    radius = _require_float(raw["radius"], f"{pfx}.radius")
    if radius <= 0.0:
        raise ConfigValidationError(
            f"[{pfx}] 'radius' must be > 0, got: {radius}"
        )

    height = _require_float(raw["height"], f"{pfx}.height")
    if height <= 0.0:
        raise ConfigValidationError(
            f"[{pfx}] 'height' must be > 0, got: {height}"
        )

    # Non-fatal warnings
    if not (world.x_min <= cx <= world.x_max):
        warnings.append(_ConfigWarning(
            f"{pfx}.center",
            f"Center x={cx} is outside world x_range {world.x_range}."
        ))
    if not (world.y_min <= cy <= world.y_max):
        warnings.append(_ConfigWarning(
            f"{pfx}.center",
            f"Center y={cy} is outside world y_range {world.y_range}."
        ))
    world_h = world.z_max - world.z_min
    if height > world_h:
        warnings.append(_ConfigWarning(
            f"{pfx}.height",
            f"height={height} m exceeds world z extent={world_h} m. "
            "Obstacle will be clamped to world bounds during rasterisation."
        ))

    return CylinderObstacle(
        id=obs_id,
        center=np.array([cx, cy]),
        radius=radius,
        height=height,
    )


def _validate_wall(
    raw: dict,
    idx: int,
    world: WorldBounds,
    warnings: List[_ConfigWarning],
) -> WallObstacle:
    obs_id = str(raw.get("id", f"wall_{idx}"))
    pfx    = f"obstacles.walls[{idx}] '{obs_id}'"

    for key in ("corners", "height"):
        if key not in raw:
            raise ConfigValidationError(f"[{pfx}] Missing required field: '{key}'")

    corners_raw = raw["corners"]
    if not (isinstance(corners_raw, (list, tuple)) and len(corners_raw) == 4):
        n = len(corners_raw) if isinstance(corners_raw, (list, tuple)) else "?"
        raise ConfigValidationError(
            f"[{pfx}] 'corners' must be a list of exactly 4 [x, y] pairs "
            f"(got {n} entries). "
            "Walls define a rectangular footprint — use exactly 4 corner points."
        )

    parsed: List[List[float]] = []
    for ci, corner in enumerate(corners_raw):
        if not (isinstance(corner, (list, tuple)) and len(corner) == 2):
            raise ConfigValidationError(
                f"[{pfx}] corners[{ci}] must be [x, y], got: {corner!r}"
            )
        parsed.append([
            _require_float(corner[0], f"{pfx}.corners[{ci}][0]"),
            _require_float(corner[1], f"{pfx}.corners[{ci}][1]"),
        ])

    corners_arr = np.array(parsed)

    # Degenerate polygon check (shoelace area)
    c = corners_arr
    area = 0.5 * abs(
        sum(
            c[i, 0] * c[(i + 1) % 4, 1] - c[(i + 1) % 4, 0] * c[i, 1]
            for i in range(4)
        )
    )
    if area < 1e-6:
        warnings.append(_ConfigWarning(
            f"{pfx}.corners",
            f"Wall footprint area is ~0 ({area:.2e} m²). "
            "The corners may be collinear — this wall will have no effect."
        ))

    height = _require_float(raw["height"], f"{pfx}.height")
    if height <= 0.0:
        raise ConfigValidationError(
            f"[{pfx}] 'height' must be > 0, got: {height}"
        )

    world_h = world.z_max - world.z_min
    if height > world_h:
        warnings.append(_ConfigWarning(
            f"{pfx}.height",
            f"height={height} m exceeds world z extent={world_h} m. "
            "Obstacle will be clamped to world bounds during rasterisation."
        ))

    return WallObstacle(id=obs_id, corners=corners_arr, height=height)


# ─────────────────────────────────────────────────────────────────────────────
# Main Environment class
# ─────────────────────────────────────────────────────────────────────────────

class Environment:
    """
    3-D planning environment with cylindrical and wall obstacles.

    Attributes
    ----------
    world     : WorldBounds
    cylinders : list[CylinderObstacle]
    walls     : list[WallObstacle]
    """

    def __init__(
        self,
        world:     WorldBounds,
        cylinders: List[CylinderObstacle],
        walls:     List[WallObstacle],
    ) -> None:
        self.world     = world
        self.cylinders = cylinders
        self.walls     = walls
        self._voxel_grid: Optional[VoxelGrid] = None  # lazy cache

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, config_path: str) -> "Environment":
        """
        Load and validate an environment from a YAML file.

        Parameters
        ----------
        config_path : str
            Path to ``environment.yaml``.

        Raises
        ------
        ConfigValidationError
            On any unrecoverable error (missing fields, wrong types, etc.).
        """
        log.debug(f"Loading environment config: '{config_path}'")

        # ── Read file ─────────────────────────────────────────────────────────
        if not os.path.isfile(config_path):
            raise ConfigValidationError(
                f"Config file not found: '{config_path}'"
            )
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigValidationError(
                f"YAML parse error in '{config_path}':\n  {exc}"
            ) from exc

        if raw is None:
            raise ConfigValidationError(
                f"Config file is empty: '{config_path}'"
            )
        if not isinstance(raw, dict):
            raise ConfigValidationError(
                f"Top-level of '{config_path}' must be a YAML mapping, "
                f"got: {type(raw).__name__}"
            )

        warnings: List[_ConfigWarning] = []

        # ── World ──────────────────────────────────────────────────────────────
        if "world" not in raw:
            raise ConfigValidationError(
                "Missing top-level 'world' section. "
                "Expected keys: x_range, y_range, z_range, resolution."
            )
        world = _validate_world(raw["world"], warnings)

        # ── Obstacles ──────────────────────────────────────────────────────────
        obs_raw = raw.get("obstacles") or {}
        if not obs_raw:
            warnings.append(_ConfigWarning(
                "obstacles",
                "No 'obstacles' section found. Environment will be empty."
            ))

        cylinders: List[CylinderObstacle] = []
        for i, c in enumerate(obs_raw.get("cylinders") or []):
            cylinders.append(_validate_cylinder(c, i, world, warnings))

        walls: List[WallObstacle] = []
        for i, w in enumerate(obs_raw.get("walls") or []):
            walls.append(_validate_wall(w, i, world, warnings))

        # ── Emit warnings ──────────────────────────────────────────────────────
        for warn in warnings:
            log.warning(f"[{warn.field}] {warn.message}")

        env = cls(world=world, cylinders=cylinders, walls=walls)

        log.debug(
            f"Loaded  world=({world.x_range}, {world.y_range}, {world.z_range})  "
            f"res={world.resolution} m/voxel  "
            f"→  grid {world.nx}×{world.ny}×{world.nz} "
            f"({world.nx * world.ny * world.nz:,} voxels)"
        )
        log.debug(
            f"Obstacles: {len(cylinders)} cylinder(s), {len(walls)} wall(s)"
        )
        return env

    # ── Analytic collision check ───────────────────────────────────────────────

    def is_point_in_obstacle(self, point: np.ndarray) -> bool:
        """
        Return True if the 3-D world-space point lies inside any obstacle.

        Uses analytic geometry — no voxel grid required. Suitable for fast
        validity checks during trajectory evaluation.
        """
        z_floor = self.world.z_min
        for cyl in self.cylinders:
            if cyl.contains_point(point, z_floor=z_floor):
                return True
        for wall in self.walls:
            if wall.contains_point(point, z_floor=z_floor):
                return True
        return False

    # ── Voxel grid (cached) ────────────────────────────────────────────────────

    def to_voxel_grid(self) -> VoxelGrid:
        """
        Rasterise all obstacles onto a Boolean voxel grid.

        The grid is built once and cached on subsequent calls.
        Voxel index (i, j, k) corresponds to the world-space cell whose
        centre is ``world.index_to_world((i, j, k))``.

        Returns
        -------
        VoxelGrid
            Compatible with ``astar_3d`` from ``traj_gen_utils``.
        """
        if self._voxel_grid is not None:
            return self._voxel_grid

        w   = self.world
        log.debug(
            f"Rasterising obstacles onto {w.nx}×{w.ny}×{w.nz} voxel grid …"
        )
        grid = np.zeros((w.nx, w.ny, w.nz), dtype=bool)

        # ── Cylinders ─────────────────────────────────────────────────────────
        for cyl in self.cylinders:
            # Tight bounding box in index space
            ix_lo = max(0, int((cyl.center[0] - cyl.radius - w.x_min) / w.resolution))
            ix_hi = min(w.nx, int((cyl.center[0] + cyl.radius - w.x_min) / w.resolution) + 2)
            iy_lo = max(0, int((cyl.center[1] - cyl.radius - w.y_min) / w.resolution))
            iy_hi = min(w.ny, int((cyl.center[1] + cyl.radius - w.y_min) / w.resolution) + 2)
            iz_hi = min(w.nz, int(cyl.height / w.resolution) + 1)

            ixs = np.arange(ix_lo, ix_hi)
            iys = np.arange(iy_lo, iy_hi)
            IX, IY = np.meshgrid(ixs, iys, indexing="ij")

            # Voxel-centre world coordinates
            vx = w.x_min + (IX + 0.5) * w.resolution
            vy = w.y_min + (IY + 0.5) * w.resolution

            in_circle = (
                (vx - cyl.center[0]) ** 2 + (vy - cyl.center[1]) ** 2
                <= cyl.radius ** 2
            )
            # Broadcast the 2-D mask along z
            grid[ix_lo:ix_hi, iy_lo:iy_hi, :iz_hi] |= in_circle[:, :, np.newaxis]

        # ── Walls ─────────────────────────────────────────────────────────────
        for wall in self.walls:
            c     = wall.corners
            cx_lo = max(0, int((c[:, 0].min() - w.x_min) / w.resolution))
            cx_hi = min(w.nx, int((c[:, 0].max() - w.x_min) / w.resolution) + 2)
            cy_lo = max(0, int((c[:, 1].min() - w.y_min) / w.resolution))
            cy_hi = min(w.ny, int((c[:, 1].max() - w.y_min) / w.resolution) + 2)
            iz_hi = min(w.nz, int(wall.height / w.resolution) + 1)

            ixs = np.arange(cx_lo, cx_hi)
            iys = np.arange(cy_lo, cy_hi)
            IX, IY = np.meshgrid(ixs, iys, indexing="ij")

            vx   = w.x_min + (IX + 0.5) * w.resolution
            vy   = w.y_min + (IY + 0.5) * w.resolution
            pts  = np.stack([vx.ravel(), vy.ravel()], axis=1)
            mask = _points_in_polygon(pts, c).reshape(IX.shape)

            grid[cx_lo:cx_hi, cy_lo:cy_hi, :iz_hi] |= mask[:, :, np.newaxis]

        # ── Wrap in VoxelGrid ─────────────────────────────────────────────────
        # origin=(0,0,0): world↔index conversion is handled by WorldBounds,
        # not by GridInfo, because GridInfo does not support float resolution.
        info = GridInfo(origin=(0, 0, 0), shape=(w.nx, w.ny, w.nz))
        self._voxel_grid = VoxelGrid(grid, info)

        occupied = int(grid.sum())
        total    = w.nx * w.ny * w.nz
        log.debug(
            f"Voxel grid ready: "
            f"{occupied:,} occupied / {total:,} total  "
            f"({100.0 * occupied / total:.2f}% density)"
        )
        return self._voxel_grid

    # ── Human-readable summary ─────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a formatted multi-line summary string."""
        w   = self.world
        sep = "─" * 64
        lines = [
            sep,
            "  ENVIRONMENT SUMMARY",
            sep,
            f"  World bounds:",
            f"    X : [{w.x_min}, {w.x_max}] m",
            f"    Y : [{w.y_min}, {w.y_max}] m",
            f"    Z : [{w.z_min}, {w.z_max}] m",
            f"    Resolution : {w.resolution} m/voxel",
            f"    Grid size  : {w.nx} × {w.ny} × {w.nz} "
            f"= {w.nx * w.ny * w.nz:,} voxels",
            "",
            f"  Cylinders ({len(self.cylinders)}):",
        ]
        for cyl in self.cylinders:
            lines.append(
                f"    [{cyl.id:20s}]  "
                f"center=({cyl.center[0]:.1f}, {cyl.center[1]:.1f})  "
                f"r={cyl.radius:.2f} m  h={cyl.height:.2f} m"
            )
        if not self.cylinders:
            lines.append("    (none)")

        lines.append(f"  Walls ({len(self.walls)}):")
        for wall in self.walls:
            c = wall.corners
            lines.append(
                f"    [{wall.id:20s}]  "
                f"corners={c.tolist()}  h={wall.height:.2f} m"
            )
        if not self.walls:
            lines.append("    (none)")

        lines.append(sep)
        return "\n".join(lines)
