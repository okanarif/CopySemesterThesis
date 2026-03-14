"""
UAV configuration dataclass and Fleet loader.

Provides:
  - UAVConfig  : immutable per-UAV mission + physical parameters
  - Fleet      : loads fleet.yaml, validates, exposes UAV list
                 and optional cross-check against an Environment
  - ConfigValidationError  (re-exported from environment)

Typical usage
-------------
    from uav import Fleet

    try:
        fleet = Fleet.from_yaml("configs/fleet.yaml")
    except ConfigValidationError as e:
        ...

    fleet.validate_with_environment(env)   # optional safety check
    print(fleet.summary())
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import yaml

# ── path bootstrap ─────────────────────────────────────────────────────────────
_swarm_dir   = os.path.dirname(os.path.abspath(__file__))
_uavsafe_dir = os.path.dirname(_swarm_dir)
for _p in (_swarm_dir, _uavsafe_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from logger      import get_logger
from environment import ConfigValidationError   # re-use shared exception

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-assigned colour palette (cycles if more UAVs than colours)
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    "#FF6B6B",  # coral
    "#4ECDC4",  # teal
    "#FFE66D",  # yellow
    "#A8E6CF",  # mint
    "#FF8B94",  # rose
    "#6C5CE7",  # violet
    "#00B894",  # emerald
    "#FDCB6E",  # orange
]

_DEFAULT_V_MAX  = 5.0
_DEFAULT_A_MAX  = 3.0
_DEFAULT_RADIUS = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Warning carrier (internal)
# ─────────────────────────────────────────────────────────────────────────────

class _W:
    __slots__ = ("field", "message")

    def __init__(self, field: str, message: str) -> None:
        self.field   = field
        self.message = message


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class UAVConfig:
    """
    Immutable configuration for a single UAV.

    Attributes
    ----------
    id     : unique string identifier
    start  : (3,) start position in world coordinates [x, y, z]
    goal   : (3,) goal  position in world coordinates [x, y, z]
    v_max  : maximum speed [m/s]
    a_max  : maximum acceleration [m/s²]
    radius : safety radius for inter-UAV collision avoidance [m]
    color  : hex colour string auto-assigned from the palette
    """
    id:     str
    start:  np.ndarray
    goal:   np.ndarray
    v_max:  float
    a_max:  float
    radius: float
    color:  str

    # numpy arrays are mutable, so frozen=True doesn't cover them;
    # we accept this and treat them as read-only by convention.
    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, UAVConfig) and self.id == other.id


# ─────────────────────────────────────────────────────────────────────────────
# YAML validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_float(value: object, path: str) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ConfigValidationError(
            f"[{path}] Expected a number, got {type(value).__name__!r}: {value!r}"
        )


def _require_xyz(value: object, path: str) -> np.ndarray:
    if not (isinstance(value, (list, tuple)) and len(value) == 3):
        raise ConfigValidationError(
            f"[{path}] Must be a list of 3 numbers [x, y, z], got: {value!r}"
        )
    return np.array([
        _require_float(value[0], f"{path}[0]"),
        _require_float(value[1], f"{path}[1]"),
        _require_float(value[2], f"{path}[2]"),
    ])


def _validate_uav(
    raw:      dict,
    idx:      int,
    defaults: dict,
    color:    str,
    warnings: List[_W],
) -> UAVConfig:
    obs_id = str(raw.get("id", f"uav_{idx}"))
    pfx    = f"uavs[{idx}] '{obs_id}'"

    for key in ("start", "goal"):
        if key not in raw:
            raise ConfigValidationError(
                f"[{pfx}] Missing required field: '{key}'"
            )

    start = _require_xyz(raw["start"], f"{pfx}.start")
    goal  = _require_xyz(raw["goal"],  f"{pfx}.goal")

    if np.allclose(start, goal):
        raise ConfigValidationError(
            f"[{pfx}] 'start' and 'goal' are the same point: {start.tolist()}"
        )

    v_max  = _require_float(raw.get("v_max",  defaults["v_max"]),  f"{pfx}.v_max")
    a_max  = _require_float(raw.get("a_max",  defaults["a_max"]),  f"{pfx}.a_max")
    radius = _require_float(raw.get("radius", defaults["radius"]), f"{pfx}.radius")

    if v_max <= 0:
        raise ConfigValidationError(f"[{pfx}] v_max must be > 0, got {v_max}")
    if a_max <= 0:
        raise ConfigValidationError(f"[{pfx}] a_max must be > 0, got {a_max}")
    if radius <= 0:
        raise ConfigValidationError(f"[{pfx}] radius must be > 0, got {radius}")

    return UAVConfig(
        id=obs_id, start=start, goal=goal,
        v_max=v_max, a_max=a_max, radius=radius,
        color=color,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fleet class
# ─────────────────────────────────────────────────────────────────────────────

class Fleet:
    """
    Collection of UAVs loaded from a YAML configuration file.

    Attributes
    ----------
    uavs : list[UAVConfig]
    """

    def __init__(self, uavs: List[UAVConfig]) -> None:
        self.uavs = uavs

    # ── construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, config_path: str) -> "Fleet":
        """
        Load and validate a fleet from a YAML file.

        The file may be either:

        * A **standalone fleet file** (the historical ``fleet.yaml`` format)
          whose top-level keys are ``defaults:`` and ``uavs:``.
        * An **environment file** that contains a nested ``fleet:`` block
          alongside ``world:`` and ``obstacles:``.  In this case the
          ``fleet:`` sub-mapping is extracted automatically.

        Raises
        ------
        ConfigValidationError
            On any unrecoverable error.
        """
        log.debug(f"Loading fleet config: '{config_path}'")

        if not os.path.isfile(config_path):
            raise ConfigValidationError(
                f"Fleet config file not found: '{config_path}'"
            )
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigValidationError(
                f"YAML parse error in '{config_path}':\n  {exc}"
            ) from exc

        if raw is None:
            raise ConfigValidationError(f"Fleet config file is empty: '{config_path}'")
        if not isinstance(raw, dict):
            raise ConfigValidationError(
                f"Top-level of '{config_path}' must be a YAML mapping."
            )

        # ── Support environment files that embed a fleet: block ───────────────
        if "fleet" in raw:
            fleet_raw = raw["fleet"]
            if not isinstance(fleet_raw, dict):
                raise ConfigValidationError(
                    f"['{config_path}'] 'fleet' key must be a YAML mapping."
                )
            raw = fleet_raw
            log.debug("  → fleet block found inside environment file")

        warnings: List[_W] = []

        # ── defaults ──────────────────────────────────────────────────────────
        def_raw  = raw.get("defaults") or {}
        defaults = {
            "v_max":  _require_float(def_raw.get("v_max",  _DEFAULT_V_MAX),  "defaults.v_max"),
            "a_max":  _require_float(def_raw.get("a_max",  _DEFAULT_A_MAX),  "defaults.a_max"),
            "radius": _require_float(def_raw.get("radius", _DEFAULT_RADIUS), "defaults.radius"),
        }

        # ── UAV list ──────────────────────────────────────────────────────────
        uavs_raw = raw.get("uavs")
        if not uavs_raw:
            raise ConfigValidationError(
                "Fleet config has no 'uavs' list. "
                "Add at least one UAV entry with 'id', 'start', and 'goal'."
            )

        uavs: List[UAVConfig] = []
        seen_ids: set = set()

        for i, u in enumerate(uavs_raw):
            color = _PALETTE[i % len(_PALETTE)]
            uav   = _validate_uav(u, i, defaults, color, warnings)

            if uav.id in seen_ids:
                raise ConfigValidationError(
                    f"[uavs[{i}]] Duplicate UAV id: '{uav.id}'"
                )
            seen_ids.add(uav.id)
            uavs.append(uav)

        # Warn if two UAVs share the same start or goal
        for i in range(len(uavs)):
            for j in range(i + 1, len(uavs)):
                if np.allclose(uavs[i].start, uavs[j].start):
                    warnings.append(_W(
                        f"uavs[{i}].start / uavs[{j}].start",
                        f"'{uavs[i].id}' and '{uavs[j].id}' share the same start position "
                        f"{uavs[i].start.tolist()} — this will cause an immediate collision."
                    ))
                if np.allclose(uavs[i].goal, uavs[j].goal):
                    warnings.append(_W(
                        f"uavs[{i}].goal / uavs[{j}].goal",
                        f"'{uavs[i].id}' and '{uavs[j].id}' share the same goal position "
                        f"{uavs[i].goal.tolist()}."
                    ))

        for w in warnings:
            log.warning(f"[{w.field}] {w.message}")

        log.debug(f"Fleet loaded: {len(uavs)} UAV(s)")
        return cls(uavs=uavs)

    # ── cross-check with environment ──────────────────────────────────────────

    def validate_with_environment(self, env) -> None:
        """
        Check that every UAV's start and goal lie inside the world bounds
        and are not inside any obstacle.

        Raises
        ------
        ConfigValidationError
            On the first invalid position found (out-of-bounds or in-obstacle).
        """
        for uav in self.uavs:
            for label, pos in (("start", uav.start), ("goal", uav.goal)):
                if not env.world.in_bounds(pos):
                    raise ConfigValidationError(
                        f"{uav.id}.{label} {pos.tolist()} is outside world bounds "
                        f"X{env.world.x_range} Y{env.world.y_range} Z{env.world.z_range}"
                    )
                if env.is_point_in_obstacle(pos):
                    raise ConfigValidationError(
                        f"{uav.id}.{label} {pos.tolist()} is inside an obstacle"
                    )

    # ── human-readable summary ────────────────────────────────────────────────

    def summary(self) -> str:
        sep   = "─" * 64
        lines = [sep, "  FLEET SUMMARY", sep,
                 f"  UAVs ({len(self.uavs)}):"]
        for uav in self.uavs:
            s = uav.start.tolist()
            g = uav.goal.tolist()
            lines.append(
                f"    [{uav.id:12s}]  "
                f"start=({s[0]:.1f}, {s[1]:.1f}, {s[2]:.1f})  "
                f"goal=({g[0]:.1f}, {g[1]:.1f}, {g[2]:.1f})  "
                f"v_max={uav.v_max:.1f}  a_max={uav.a_max:.1f}  r={uav.radius:.2f}"
            )
        lines.append(sep)
        return "\n".join(lines)
